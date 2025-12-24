# pyright: reportPrivateImportUsage=false
# pyright: reportAttributeAccessIssue=false
import os
import torch
import torch.nn.functional as F
import math
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, cast
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.training_utils import cast_training_params
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from safetensors.torch import save_file
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from .schema import TrainingConfig
from .engine_utils import (
    OnikaDataset, 
    get_optimizer, 
    apply_optimizations, 
    flush, 
    generate_samples, 
    save_lora_weights, 
    collate_fn_general,
    inject_network,
    save_model_card,
    manage_checkpoints,
    cache_latents_to_disk,
    get_latents_cache_dir
)
from .loss_utils import (
    conditional_loss,
    apply_noise_offset,
    apply_input_perturbation,
)

def train_flux(config: TrainingConfig, status_callback: Callable):
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def compute_density_for_timestep_sampling(
        weighting_scheme: str, batch_size: int, logit_mean: Optional[float] = None, logit_std: Optional[float] = None, mode_scale: Optional[float] = None
    ):
        if weighting_scheme == "logit_normal":
            u = torch.normal(mean=logit_mean or 0.0, std=logit_std or 1.0, size=(batch_size,), device=accelerator.device)
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device=accelerator.device)
            u = 1 - u - (mode_scale or 0.0) * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device=accelerator.device)
        return u

    # 1. Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    
    if config.seed is not None:
        set_seed(config.seed)

    # Auto-generate reg images if needed
    from .engine_utils import generate_class_images
    generate_class_images(config, accelerator, status_callback)

    # 2. Load Models
    status_callback(0, 0, 0, 0, f"Loading Flux model from {config.base_model_name}...")
    
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float32
    }
    if config.quantization_bit == "4":
        load_kwargs["load_in_4bit"] = True
    elif config.quantization_bit == "8":
        load_kwargs["load_in_8bit"] = True

    pipeline = FluxPipeline.from_single_file(
        config.base_model_name,
        **load_kwargs
    )
    
    transformer = pipeline.transformer
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    noise_scheduler = cast(FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config))
    
    # We'll keep the pipeline around for encoding if needed, or just use its components
    # del pipeline
    # flush()
    
    # 3. Prepare Models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    
    apply_optimizations(transformer, config)
    
    # 4. Inject Network (LoRA, LoHa, LoKr, etc.)
    status_callback(0, 0, 0, 0, f"Injecting {config.network_type}...")
    
    # Network setup
    if config.lora_layers:
        target_modules = [layer.strip() for layer in config.lora_layers.split(",")]
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    
    if config.lora_blocks:
        target_blocks = [int(block.strip()) for block in config.lora_blocks.split(",")]
        new_target_modules = []
        for block in target_blocks:
            for module in target_modules:
                new_target_modules.append(f"double_blocks.{block}.img_attn.{module}")
                new_target_modules.append(f"double_blocks.{block}.txt_attn.{module}")
                new_target_modules.append(f"single_blocks.{block}.attn.{module}")
        target_modules = new_target_modules

    transformer = inject_network(transformer, config, target_modules)

    if config.train_text_encoder:
        te_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        text_encoder = inject_network(text_encoder, config, te_target_modules, is_text_encoder=True)
        text_encoder_2 = inject_network(text_encoder_2, config, te_target_modules, is_text_encoder=True)

    # Cast trainable params to float32
    if accelerator.mixed_precision == "fp16":
        models = [transformer]
        if config.train_text_encoder:
            models.extend([text_encoder, text_encoder_2])
        cast_training_params(models, dtype=torch.float32)

    # 5. Optimizer
    params_to_optimize = [{"params": list(filter(lambda p: p.requires_grad, transformer.parameters())), "lr": config.learning_rate}]
    if config.train_text_encoder:
        params_to_optimize.append({
            "params": list(filter(lambda p: p.requires_grad, text_encoder.parameters())),
            "lr": config.text_encoder_lr or config.learning_rate
        })
        params_to_optimize.append({
            "params": list(filter(lambda p: p.requires_grad, text_encoder_2.parameters())),
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
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.dataloader_shuffle,
        collate_fn=collate_fn_general,
        num_workers=config.dataloader_num_workers
    )
    
    # 7. Prepare with Accelerator
    if config.train_text_encoder:
        transformer, text_encoder, text_encoder_2, optimizer, dataloader = accelerator.prepare(
            transformer, text_encoder, text_encoder_2, optimizer, dataloader
        )
    else:
        transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)

    # Potentially resume from checkpoint
    global_step = 0
    first_epoch = 0
    resume_step = 0
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        else:
            path = os.path.basename(config.resume_from_checkpoint)

        if path is None:
            if status_callback:
                status_callback(0, 0, 0, 0, f"Checkpoint '{config.resume_from_checkpoint}' not found. Starting from scratch.")
        else:
            if status_callback:
                status_callback(0, 0, 0, 0, f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    # 8. Training Loop
    num_update_steps_per_epoch = len(dataloader) // config.gradient_accumulation_steps
    max_train_steps = config.max_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # Custom saving & loading hooks
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            save_lora_weights(accelerator, transformer, config, global_step,
                              text_encoder if config.train_text_encoder else None,
                              text_encoder_2 if config.train_text_encoder else None)

    def load_model_hook(models, input_dir):
        pass

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    global_step = 0
    first_epoch = 0
    resume_step = 0
    
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
            resume_step = global_step % num_update_steps_per_epoch
            status_callback(global_step, max_train_steps, 0, first_epoch, f"Resumed from {path}")
    
    # Pre-compute text embeddings if requested
    cached_embeddings = None
    if config.cache_text_embeddings:
        status_callback(0, 0, 0, 0, "Caching text embeddings...")
        cached_embeddings = {}
        for i, batch in enumerate(tqdm(dataloader, desc="Caching embeddings")):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                    batch["prompts"],
                    prompt_2=batch["prompts"],
                    max_sequence_length=config.max_sequence_length or 512
                )
                cached_embeddings[i] = (prompt_embeds.cpu(), pooled_prompt_embeds.cpu(), text_ids.cpu())
        
        if not config.train_text_encoder:
            text_encoder.to("cpu")
            text_encoder_2.to("cpu")
            flush()

    status_callback(global_step, max_train_steps, 0, first_epoch, "Starting training loop...")
    
    for epoch in range(first_epoch, config.max_train_epochs):
        transformer.train()
        if config.train_text_encoder:
            text_encoder.train()
            text_encoder_2.train()
            
        for step, batch in enumerate(dataloader):
            # Skip steps until we reach the resumed step
            if config.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            models_to_accumulate = [transformer]
            if config.train_text_encoder:
                models_to_accumulate.extend([text_encoder, text_encoder_2])
                
            with accelerator.accumulate(models_to_accumulate):
                # Latents
                if "latents" in batch:
                    latents = batch["latents"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)
                
                # Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Flux uses flow matching with density sampling
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=config.logit_mean,
                    logit_std=config.logit_std,
                    mode_scale=config.mode_scale,
                )
                num_train_timesteps = int(noise_scheduler.config.num_train_timesteps)
                min_ts = 0 if config.min_timestep is None else int(config.min_timestep)
                max_ts = (num_train_timesteps - 1) if config.max_timestep is None else int(config.max_timestep)
                min_ts = max(0, min_ts)
                max_ts = min(num_train_timesteps - 1, max_ts)
                if min_ts > max_ts:
                    min_ts, max_ts = 0, num_train_timesteps - 1

                u_cpu = u.detach().cpu() if hasattr(u, "detach") else u
                range_size = max_ts - min_ts + 1
                indices = (u_cpu * range_size).long() + min_ts
                indices = indices.clamp(min_ts, max_ts)
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                # Add noise (Flow Matching)
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                
                # Pack latents
                packed_noisy_latents = FluxPipeline._pack_latents(
                    noisy_latents,
                    batch_size=bsz,
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )
                
                # Image IDs
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    bsz,
                    latents.shape[2] // 2,
                    latents.shape[3] // 2,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )

                # Prompts
                if cached_embeddings and step in cached_embeddings:
                    prompt_embeds, pooled_prompt_embeds, text_ids = cached_embeddings[step]
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                    text_ids = text_ids.to(accelerator.device)
                else:
                    with torch.set_grad_enabled(config.train_text_encoder):
                        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                            batch["prompts"],
                            prompt_2=batch["prompts"],
                            max_sequence_length=config.max_sequence_length or 512
                        )

                # Predict
                model_pred = transformer(
                    hidden_states=packed_noisy_latents,
                    timestep=timesteps / 1000.0,
                    guidance=torch.tensor([config.guidance_scale], device=accelerator.device).expand(bsz),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                ).sample
                
                # Loss for flow matching: pred should be (noise - latents)
                target = noise - latents
                
                if config.use_prior_preservation:
                    # Chunk the model_pred and target into two parts, the first for the conditioning images and the second for the class images
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss
                    loss = loss + config.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                status_callback(global_step, max_train_steps, loss.item(), epoch, None)
                
                # Save every n steps
                if config.save_every_n_steps and global_step % config.save_every_n_steps == 0:
                    # save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    # accelerator.save_state(save_path)
                    # manage_checkpoints(config.output_dir, config.checkpoints_total_limit)
                    save_lora_weights(accelerator, transformer, config, global_step,
                                      text_encoder if config.train_text_encoder else None,
                                      text_encoder_2 if config.train_text_encoder else None)

                # Sample generation
                if config.sample_every_n_steps and global_step % config.sample_every_n_steps == 0:
                    if not config.train_text_encoder and config.cache_text_embeddings:
                        text_encoder.to(accelerator.device)
                        text_encoder_2.to(accelerator.device)
                        
                    sampling_pipeline = FluxPipeline(
                        transformer=accelerator.unwrap_model(transformer),
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_2),
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        scheduler=noise_scheduler,
                    ).to(accelerator.device)
                    generate_samples(sampling_pipeline, config, global_step, epoch)
                    del sampling_pipeline
                    
                    if not config.train_text_encoder and config.cache_text_embeddings:
                        text_encoder.to("cpu")
                        text_encoder_2.to("cpu")
                    flush()

        # Epoch end
        if config.sample_every_n_epochs and (epoch + 1) % config.sample_every_n_epochs == 0:
            if not config.train_text_encoder and config.cache_text_embeddings:
                text_encoder.to(accelerator.device)
                text_encoder_2.to(accelerator.device)
                
            sampling_pipeline = FluxPipeline(
                transformer=accelerator.unwrap_model(transformer),
                vae=vae,
                text_encoder=accelerator.unwrap_model(text_encoder),
                text_encoder_2=accelerator.unwrap_model(text_encoder_2),
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                scheduler=noise_scheduler,
            ).to(accelerator.device)
            generate_samples(sampling_pipeline, config, global_step, epoch)
            del sampling_pipeline
            
            if not config.train_text_encoder and config.cache_text_embeddings:
                text_encoder.to("cpu")
                text_encoder_2.to("cpu")
            flush()

        if (epoch + 1) % config.save_every_n_epochs == 0:
            # save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
            # accelerator.save_state(save_path)
            # manage_checkpoints(config.output_dir, config.checkpoints_total_limit)
            save_lora_weights(accelerator, transformer, config, global_step,
                              text_encoder if config.train_text_encoder else None,
                              text_encoder_2 if config.train_text_encoder else None)

    # 9. Save Final
    status_callback(global_step, max_train_steps, 0, 0, "Saving Final LoRA...")
    save_lora_weights(accelerator, transformer, config, -1,
                      text_encoder if config.train_text_encoder else None,
                      text_encoder_2 if config.train_text_encoder else None)
    
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
        del transformer, vae, text_encoder, text_encoder_2, pipeline, noise_scheduler
        del optimizer, lr_scheduler, train_dataloader, accelerator
    except:
        pass
    flush()

