# pyright: reportPrivateImportUsage=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportOptionalOperand=false
# pyright: reportOperatorIssue=false
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Callable, List, Dict, Any, cast
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr, cast_training_params
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from safetensors.torch import save_file
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from .schema import TrainingConfig
from .engine_utils import (
    OnikaDataset, 
    get_optimizer, 
    apply_optimizations, 
    generate_samples,
    save_lora_weights,
    flush,
    collate_fn_sdxl,
    inject_network,
    save_model_card,
    manage_checkpoints,
    cache_latents_to_disk,
    get_latents_cache_dir
)
from .loss_utils import (
    conditional_loss,
    apply_snr_weight,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_noise_offset,
    apply_input_perturbation,
    fix_noise_scheduler_betas_for_zero_terminal_snr,
    scale_v_prediction_loss_like_noise_prediction,
)

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids

def encode_prompt(text_encoders, tokenizers, prompt, proportion_empty_prompts=0.0, is_train=True):
    prompt_embeds_list = []
    pooled_prompt_embeds = None
    
    for i, text_encoder in enumerate(text_encoders):
        tokenizer = tokenizers[i]
        text_input_ids = tokenize_prompt(tokenizer, prompt)
        
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )
        
        # We are only interested in the pooled output of the second text encoder
        if i == 1:
            pooled_prompt_embeds = prompt_embeds[0]
        
        # The last hidden state is the second to last hidden state
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds

def train_sdxl(config: TrainingConfig, status_callback: Callable):
    # 1. Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    
    if config.seed is not None:
        set_seed(config.seed)

    # 2. Load Models
    status_callback(0, 0, 0, 0, f"Loading SDXL model from {config.base_model_name}...")
    
    # Auto-generate reg images if needed
    from .engine_utils import generate_class_images
    generate_class_images(config, accelerator, status_callback)
    
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float32
    }
    if config.quantization_bit == "4":
        load_kwargs["load_in_4bit"] = True
    elif config.quantization_bit == "8":
        load_kwargs["load_in_8bit"] = True

    pipeline = StableDiffusionXLPipeline.from_single_file(
        config.base_model_name,
        **load_kwargs
    )
    
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder_one = pipeline.text_encoder
    text_encoder_two = pipeline.text_encoder_2
    tokenizer_one = pipeline.tokenizer
    tokenizer_two = pipeline.tokenizer_2
    
    # Determine scheduler
    if config.do_edm_style_training:
        noise_scheduler = cast(EDMEulerScheduler, EDMEulerScheduler.from_config(pipeline.scheduler.config))
    else:
        noise_scheduler = cast(DDPMScheduler, DDPMScheduler.from_config(pipeline.scheduler.config))
    
    # Apply zero terminal SNR fix if enabled
    if config.zero_terminal_snr and not config.do_edm_style_training:
        fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
    
    del pipeline
    flush()
    
    # 3. Prepare Models
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    vae_dtype = torch.float32
    if not bool(getattr(config, "no_half_vae", False)) and (bool(getattr(config, "full_fp16", False)) or bool(getattr(config, "full_bf16", False))):
        vae_dtype = weight_dtype
    vae.to(accelerator.device, dtype=vae_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    apply_optimizations(unet, config)
    
    # 4. Inject Network (LoRA, LoHa, LoKr, etc.)
    status_callback(0, 0, 0, 0, f"Injecting {config.network_type}...")
    
    # Network setup
    if config.lora_layers:
        target_modules = [layer.strip() for layer in config.lora_layers.split(",")]
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    
    if config.lora_blocks:
        target_blocks = [block.strip() for block in config.lora_blocks.split(",")]
        new_target_modules = []
        for block in target_blocks:
            for module in target_modules:
                # SDXL UNet structure: down_blocks, up_blocks, mid_block
                if block.startswith("down"):
                    new_target_modules.append(f"down_blocks.{block.split('_')[1]}.{module}")
                elif block.startswith("up"):
                    new_target_modules.append(f"up_blocks.{block.split('_')[1]}.{module}")
                elif block == "mid":
                    new_target_modules.append(f"mid_block.{module}")
        target_modules = new_target_modules

    unet = inject_network(unet, config, target_modules)

    if config.train_text_encoder:
        te_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        text_encoder_one = inject_network(text_encoder_one, config, te_target_modules, is_text_encoder=True)
        text_encoder_two = inject_network(text_encoder_two, config, te_target_modules, is_text_encoder=True)

    # Cast trainable params to float32
    if accelerator.mixed_precision == "fp16":
        models = [unet]
        if config.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    # 5. Optimizer
    params_to_optimize = [{"params": list(filter(lambda p: p.requires_grad, unet.parameters())), "lr": config.learning_rate}]
    if config.train_text_encoder:
        params_to_optimize.append({
            "params": list(filter(lambda p: p.requires_grad, text_encoder_one.parameters())),
            "lr": config.text_encoder_lr or config.learning_rate
        })
        params_to_optimize.append({
            "params": list(filter(lambda p: p.requires_grad, text_encoder_two.parameters())),
            "lr": config.text_encoder_lr or config.learning_rate
        })

    optimizer = get_optimizer(config, params_to_optimize)
    
    # 6. Dataset
    dataset = OnikaDataset(config)
    
    # Cache Latents if requested
    if config.cache_latents_to_disk:
        status_callback(0, 0, 0, 0, "Caching latents to disk...")
        cache_latents_to_disk(vae, dataset, config, accelerator, status_callback=status_callback)
        dataset.use_cached_latents = True
        dataset.cache_dir = get_latents_cache_dir(config)
        
        # Move VAE to CPU to save VRAM since we have cached latents
        vae.to("cpu")
        flush()
    
    persistent_workers = bool(getattr(config, "persistent_data_loader_workers", False))
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.dataloader_shuffle,
        collate_fn=collate_fn_sdxl,
        num_workers=config.dataloader_num_workers,
        persistent_workers=persistent_workers if int(getattr(config, "dataloader_num_workers", 0) or 0) > 0 else False,
    )
    
    # 7. Prepare with Accelerator
    if config.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, dataloader = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, dataloader
        )
    else:
        unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    # 8. Training Loop
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def per_element_loss(pred, tgt):
        lt = (getattr(config, "loss_type", None) or "l2").strip().lower()
        if lt in {"l1", "mae"}:
            return F.l1_loss(pred.float(), tgt.float(), reduction="none")
        if lt == "huber":
            return F.huber_loss(pred.float(), tgt.float(), reduction="none", delta=float(getattr(config, "huber_c", 0.1) or 0.1))
        if lt in {"smooth_l1", "smoothl1"}:
            return F.smooth_l1_loss(pred.float(), tgt.float(), reduction="none")
        return F.mse_loss(pred.float(), tgt.float(), reduction="none")

    num_update_steps_per_epoch = len(dataloader) // config.gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, int(num_update_steps_per_epoch))

    max_train_steps = int(config.max_train_steps) if config.max_train_steps else (config.max_train_epochs * num_update_steps_per_epoch)
    max_train_steps = max(1, int(max_train_steps))
    planned_epochs = int((max_train_steps + num_update_steps_per_epoch - 1) // num_update_steps_per_epoch)
    planned_epochs = max(1, planned_epochs)
    
    warmup_steps = int(config.lr_warmup_steps or 0)
    if warmup_steps <= 0 and (config.lr_warmup_ratio or 0.0) > 0:
        warmup_steps = int(max_train_steps * float(config.lr_warmup_ratio))
    warmup_steps = max(0, int(warmup_steps))

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=int(config.lr_scheduler_num_cycles or 1),
        power=float(config.lr_scheduler_power or 1.0),
    )
    
    # Custom saving & loading hooks
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # We save the LoRA weights in Kohya format inside the checkpoint folder
            # This is what accelerator.save_state calls
            save_lora_weights(accelerator, unet, config, global_step, 
                              text_encoder_one if config.train_text_encoder else None,
                              text_encoder_two if config.train_text_encoder else None)

    def load_model_hook(models, input_dir):
        # PEFT handles loading automatically if we use accelerator.load_state
        pass

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        else:
            path = config.resume_from_checkpoint

        if path:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            status_callback(global_step, max_train_steps, 0, first_epoch, f"Resumed from {path}")
    
    # Pre-compute text embeddings if requested
    cached_embeddings = None
    if config.cache_text_embeddings:
        status_callback(0, 0, 0, 0, "Caching text embeddings...")
        cached_embeddings = {}
        for i, batch in enumerate(tqdm(dataloader, desc="Caching embeddings")):
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                [text_encoder_one, text_encoder_two],
                [tokenizer_one, tokenizer_two],
                batch["prompts"]
            )
            cached_embeddings[i] = (prompt_embeds.cpu(), pooled_prompt_embeds.cpu())
        
        # Move text encoders to CPU to save VRAM if not training them
        if not config.train_text_encoder:
            text_encoder_one.to("cpu")
            text_encoder_two.to("cpu")
            flush()

    status_callback(global_step, max_train_steps, 0, first_epoch, "Starting training loop...")
    
    stop_training = False
    best_loss = float("inf")
    for epoch in range(first_epoch, planned_epochs):
        unet.train()
        if config.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
            
        for step, batch in enumerate(dataloader):
            models_to_accumulate = [unet]
            if config.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one, text_encoder_two])

            with accelerator.accumulate(models_to_accumulate):
                # Latents
                if "latents" in batch:
                    latents = batch["latents"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)
                
                # Noise with optional offset and input perturbation
                noise = torch.randn_like(latents)
                
                # Apply noise offset (improves dark/light training)
                if config.noise_offset_strength > 0 or config.noise_offset_random_strength > 0:
                    noise = apply_noise_offset(
                        noise,
                        noise_offset_type=getattr(config, "noise_offset_type", "original"),
                        noise_offset=config.noise_offset_strength,
                        noise_offset_random_strength=config.noise_offset_random_strength,
                        adaptive_noise_scale=config.adaptive_noise_scale if config.adaptive_noise_scale > 0 else None,
                        latents=latents if config.adaptive_noise_scale > 0 else None,
                    )
                
                # Apply input perturbation (regularization)
                if config.ip_noise_gamma > 0:
                    noise = apply_input_perturbation(noise, config.ip_noise_gamma)
                
                bsz = latents.shape[0]
                
                # Timesteps
                num_train_timesteps = int(noise_scheduler.config.num_train_timesteps)
                min_ts = 0 if config.min_timestep is None else int(config.min_timestep)
                max_ts = (num_train_timesteps - 1) if config.max_timestep is None else int(config.max_timestep)
                min_ts = max(0, min_ts)
                max_ts = min(num_train_timesteps - 1, max_ts)
                if min_ts > max_ts:
                    min_ts, max_ts = 0, num_train_timesteps - 1

                if not config.do_edm_style_training:
                    timesteps = torch.randint(min_ts, max_ts + 1, (bsz,), device=latents.device).long()
                else:
                    indices = torch.randint(min_ts, max_ts + 1, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # EDM Preconditioning
                sigmas = None  # Initialize for non-EDM case
                if config.do_edm_style_training:
                    sigmas = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    if "EDM" in str(type(noise_scheduler)):
                        inp_noisy_latents = noise_scheduler.precondition_inputs(noisy_latents, sigmas)
                    else:
                        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
                else:
                    inp_noisy_latents = noisy_latents

                # Prompts
                if cached_embeddings and step in cached_embeddings:
                    prompt_embeds, pooled_prompt_embeds = cached_embeddings[step]
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                else:
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        [text_encoder_one, text_encoder_two],
                        [tokenizer_one, tokenizer_two],
                        batch["prompts"]
                    )
                
                # Time IDs
                def compute_time_ids(original_size, crop_top_left):
                    target_size = (config.resolution, config.resolution)
                    add_time_ids = list(original_size + crop_top_left + target_size)
                    return torch.tensor([add_time_ids])

                add_time_ids = torch.cat([
                    compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                ]).to(accelerator.device, dtype=weight_dtype)

                # Predict
                model_pred = unet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                ).sample
                
                # EDM Output Preconditioning
                weighting = None
                if config.do_edm_style_training:
                    if "EDM" in str(type(noise_scheduler)):
                        model_pred = noise_scheduler.precondition_outputs(noisy_latents, model_pred, sigmas)
                    else:
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = model_pred * (-sigmas) + noisy_latents
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                                noisy_latents / (sigmas**2 + 1)
                            )
                    if "EDM" not in str(type(noise_scheduler)):
                        weighting = (sigmas**-2.0).float()

                # Target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = latents if config.do_edm_style_training else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = latents if config.do_edm_style_training else noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                # Loss
                if config.do_edm_style_training:
                    if weighting is not None:
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                            dim=1
                        ).mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif (config.snr_gamma or config.min_snr_gamma) and (config.snr_gamma or config.min_snr_gamma) > 0:
                    snr_gamma = config.snr_gamma if config.snr_gamma is not None else config.min_snr_gamma
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights + 1
                    
                    loss = per_element_loss(model_pred, target)
                    loss = loss.mean(dim=list(range(1, len(loss.shape))))
                    loss = loss * mse_loss_weights

                    if config.v_pred_like_loss and config.v_pred_like_loss > 0 and noise_scheduler.config.prediction_type == "epsilon":
                        snr_clamped = torch.minimum(snr, torch.ones_like(snr) * 1000)
                        scale = snr_clamped / (snr_clamped + 1)
                        loss = loss + (loss / scale) * config.v_pred_like_loss

                    loss = loss.mean()
                else:
                    loss = per_element_loss(model_pred, target)
                    loss = loss.mean(dim=list(range(1, len(loss.shape))))

                    if config.v_pred_like_loss and config.v_pred_like_loss > 0 and noise_scheduler.config.prediction_type == "epsilon":
                        snr = compute_snr(noise_scheduler, timesteps)
                        snr_clamped = torch.minimum(snr, torch.ones_like(snr) * 1000)
                        scale = snr_clamped / (snr_clamped + 1)
                        loss = loss + (loss / scale) * config.v_pred_like_loss

                    if config.use_prior_preservation:
                        loss, prior_loss = torch.chunk(loss, 2, dim=0)
                        loss = loss.mean() + config.prior_loss_weight * prior_loss.mean()
                    else:
                        loss = loss.mean()
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                status_callback(global_step, max_train_steps, loss.item(), epoch, None)

                if global_step >= max_train_steps:
                    stop_training = True
                    break
                
                # Save every n steps
                if config.save_every_n_steps and global_step % config.save_every_n_steps == 0:
                    cur_loss = float(loss.item())
                    if (not getattr(config, "save_best_only", False)) or (cur_loss < best_loss):
                        best_loss = min(best_loss, cur_loss)
                        save_lora_weights(
                            accelerator, unet, config, global_step,
                            text_encoder_one if config.train_text_encoder else None,
                            text_encoder_two if config.train_text_encoder else None,
                        )
                        manage_checkpoints(config.output_dir, config.checkpoints_total_limit, output_name=config.output_name)

                # Sample generation
                if config.sample_every_n_steps and global_step % config.sample_every_n_steps == 0:
                    # Ensure text encoders are on device for sampling
                    if not config.train_text_encoder and config.cache_text_embeddings:
                        text_encoder_one.to(accelerator.device)
                        text_encoder_two.to(accelerator.device)
                        
                    pipeline = StableDiffusionXLPipeline(
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        tokenizer=tokenizer_one,
                        tokenizer_2=tokenizer_two,
                        unet=accelerator.unwrap_model(unet),
                        scheduler=noise_scheduler,
                    )
                    generate_samples(pipeline, config, global_step, epoch)
                    del pipeline
                    
                    if not config.train_text_encoder and config.cache_text_embeddings:
                        text_encoder_one.to("cpu")
                        text_encoder_two.to("cpu")
                    flush()

            if stop_training:
                break

        # Epoch end
        if config.sample_every_n_epochs and (epoch + 1) % config.sample_every_n_epochs == 0:
            if not config.train_text_encoder and config.cache_text_embeddings:
                text_encoder_one.to(accelerator.device)
                text_encoder_two.to(accelerator.device)
                
            pipeline = StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=accelerator.unwrap_model(text_encoder_one),
                text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                tokenizer=tokenizer_one,
                tokenizer_2=tokenizer_two,
                unet=accelerator.unwrap_model(unet),
                scheduler=noise_scheduler,
            )
            generate_samples(pipeline, config, global_step, epoch)
            del pipeline
            
            if not config.train_text_encoder and config.cache_text_embeddings:
                text_encoder_one.to("cpu")
                text_encoder_two.to("cpu")
            flush()

        if (epoch + 1) % config.save_every_n_epochs == 0:
            cur_loss = float(loss.item())
            if (not getattr(config, "save_best_only", False)) or (cur_loss < best_loss):
                best_loss = min(best_loss, cur_loss)
                save_lora_weights(
                    accelerator, unet, config, global_step,
                    text_encoder_one if config.train_text_encoder else None,
                    text_encoder_two if config.train_text_encoder else None,
                )
                manage_checkpoints(config.output_dir, config.checkpoints_total_limit, output_name=config.output_name)

    # 9. Save Final
    status_callback(global_step, max_train_steps, 0, 0, "Saving Final LoRA...")
    save_lora_weights(accelerator, unet, config, -1, 
                      text_encoder_one if config.train_text_encoder else None,
                      text_encoder_two if config.train_text_encoder else None)
    
    # Save model card
    save_model_card(
        config,
        config.output_dir,
        base_model=config.base_model_name,
        train_text_encoder=config.train_text_encoder,
    )
    
    status_callback(global_step, max_train_steps, 0, 0, f"Training completed.")

    # 10. Cleanup
    try:
        del unet, vae, text_encoder_one, text_encoder_two, pipeline, noise_scheduler
        del optimizer, lr_scheduler, train_dataloader, accelerator
    except:
        pass
    flush()

