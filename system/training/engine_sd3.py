# pyright: reportPrivateImportUsage=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
import os
import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any
from accelerate import Accelerator
from diffusers import (
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModelWithProjection, T5EncoderModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from tqdm.auto import tqdm
from pathlib import Path

from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    cast_training_params,
)

from .schema import TrainingConfig
from .engine_utils import (
    OnikaDataset, 
    get_optimizer, 
    apply_optimizations, 
    generate_samples,
    save_lora_weights,
    flush,
    collate_fn_general,
    inject_network,
    save_model_card,
    manage_checkpoints,
    cache_latents_to_disk,
    get_latents_cache_dir
)

def train_sd3(config: TrainingConfig, status_callback: Optional[Callable] = None):
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    if status_callback:
        status_callback(0, 0, 0, 0, f"Loading SD3 model from {config.base_model_name}...")

    # Auto-generate reg images if needed
    from .engine_utils import generate_class_images
    generate_class_images(config, accelerator, status_callback)

    # Load models using from_single_file
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float32
    }
    if config.quantization_bit == "4":
        load_kwargs["load_in_4bit"] = True
    elif config.quantization_bit == "8":
        load_kwargs["load_in_8bit"] = True

    pipeline = StableDiffusion3Pipeline.from_single_file(
        config.base_model_name,
        **load_kwargs
    )

    transformer = pipeline.transformer
    vae = pipeline.vae
    text_encoder_one = pipeline.text_encoder
    text_encoder_two = pipeline.text_encoder_2
    text_encoder_three = pipeline.text_encoder_3
    tokenizer_one = pipeline.tokenizer
    tokenizer_two = pipeline.tokenizer_2
    tokenizer_three = pipeline.tokenizer_3
    
    # Scheduler setup
    scheduler_config = pipeline.scheduler.config
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    # Pylance might think this returns a tuple, so we can assert or cast if needed, 
    # but usually from_config returns the object. 
    # If return_unused_kwargs is True, it returns (scheduler, unused_kwargs).
    if isinstance(noise_scheduler, tuple):
        noise_scheduler = noise_scheduler[0]

    del pipeline
    flush()

    # Freeze models
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    transformer.requires_grad_(False)

    # Move to device
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
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    apply_optimizations(transformer, config)

    # Network setup
    if config.lora_layers:
        target_modules = [layer.strip() for layer in config.lora_layers.split(",")]
    else:
        target_modules = ["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out", "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"]
    
    if config.lora_blocks:
        target_blocks = [int(block.strip()) for block in config.lora_blocks.split(",")]
        target_modules = [
            f"transformer_blocks.{block}.{module}" for block in target_blocks for module in target_modules
        ]

    transformer = inject_network(transformer, config, target_modules)

    if config.train_text_encoder:
        te_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        text_encoder_one = inject_network(text_encoder_one, config, te_target_modules, is_text_encoder=True)
        text_encoder_two = inject_network(text_encoder_two, config, te_target_modules, is_text_encoder=True)

    # Cast trainable params to float32
    if accelerator.mixed_precision == "fp16":
        models = [transformer]
        if config.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    # Optimizer
    params_to_optimize = [{"params": list(filter(lambda p: p.requires_grad, transformer.parameters())), "lr": config.learning_rate}]
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

    # Dataset
    dataset = OnikaDataset(config)

    # Cache Latents if requested
    if config.cache_latents_to_disk:
        if status_callback:
            status_callback(0, 0, 0, 0, "Caching latents to disk...")
        cache_latents_to_disk(vae, dataset, config, accelerator, status_callback=status_callback, epoch=0)
        dataset.use_cached_latents = True
        dataset.cache_dir = get_latents_cache_dir(config)

        # Move VAE to CPU to save VRAM since we have cached latents
        vae.to("cpu")
        flush()
    persistent_workers = bool(getattr(config, "persistent_data_loader_workers", False))
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.dataloader_shuffle,
        collate_fn=collate_fn_general,
        num_workers=config.dataloader_num_workers,
        persistent_workers=persistent_workers if int(getattr(config, "dataloader_num_workers", 0) or 0) > 0 else False,
    )

    # Scheduler
    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    max_train_steps = config.max_train_steps or (config.max_train_epochs * num_update_steps_per_epoch)

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
            save_lora_weights(accelerator, transformer, config, global_step, 
                              text_encoder_one if config.train_text_encoder else None,
                              text_encoder_two if config.train_text_encoder else None,
                              text_encoder_three if config.train_text_encoder else None)

    def load_model_hook(models, input_dir):
        pass

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

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
            path = config.resume_from_checkpoint

        if path:
            if status_callback:
                status_callback(0, 0, 0, 0, f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype) # type: ignore
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device) # type: ignore
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def encode_sd3_prompt(prompts, max_sequence_length=256):
        with torch.set_grad_enabled(config.train_text_encoder):
            # CLIP 1
            tokens_1 = tokenizer_one(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(accelerator.device)
            out_1 = text_encoder_one(tokens_1.input_ids, output_hidden_states=True)
            prompt_embeds_1 = out_1.hidden_states[-2]
            pooled_embeds_1 = out_1[0]
            
            # CLIP 2
            tokens_2 = tokenizer_two(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(accelerator.device)
            out_2 = text_encoder_two(tokens_2.input_ids, output_hidden_states=True)
            prompt_embeds_2 = out_2.hidden_states[-2]
            pooled_embeds_2 = out_2[0]
        
        # T5 is never trained in this script
        with torch.no_grad():
            # T5
            tokens_3 = tokenizer_three(prompts, padding="max_length", max_length=max_sequence_length, truncation=True, return_tensors="pt").to(accelerator.device)
            prompt_embeds_3 = text_encoder_three(tokens_3.input_ids)[0]
        
        clip_prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        pooled_prompt_embeds = torch.cat([pooled_embeds_1, pooled_embeds_2], dim=-1)
        
        # Pad CLIP embeds to match T5 sequence length
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, prompt_embeds_3.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, prompt_embeds_3], dim=-2)
        
        return prompt_embeds, pooled_prompt_embeds

    # Pre-compute text embeddings if requested
    cached_embeddings = None
    if config.cache_text_embeddings:
        if status_callback:
            status_callback(0, 0, 0, 0, "Caching text embeddings...")
        cached_embeddings = {}
        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching embeddings")):
            prompt_embeds, pooled_embeds = encode_sd3_prompt(batch["prompts"], max_sequence_length=config.max_sequence_length or 256)
            cached_embeddings[i] = (prompt_embeds.cpu(), pooled_embeds.cpu())
        
        if not config.train_text_encoder:
            text_encoder_one.to("cpu")
            text_encoder_two.to("cpu")
            text_encoder_three.to("cpu")
            flush()

    # Training loop
    num_epochs = config.max_train_epochs
    epoch = first_epoch

    best_loss = float("inf")
    
    # Check if augmentations are enabled for epoch-based recaching
    has_augmentations = (
        getattr(config, "crop_jitter", 0.0) > 0 or
        getattr(config, "random_flip", 0.0) > 0 or
        getattr(config, "random_brightness", 0.0) > 0 or
        getattr(config, "random_contrast", 0.0) > 0 or
        getattr(config, "random_saturation", 0.0) > 0 or
        getattr(config, "random_hue", 0.0) > 0
    )

    for epoch in range(first_epoch, num_epochs):
        # Re-cache latents each epoch if disk caching + augmentations are enabled
        if config.cache_latents_to_disk and has_augmentations and epoch > first_epoch:
            status_callback(global_step, max_train_steps, 0, epoch, f"Regenerating augmented latents for epoch {epoch}...")
            vae.to(accelerator.device)
            cache_latents_to_disk(vae, dataset, config, accelerator, status_callback=status_callback, epoch=epoch)
            vae.to("cpu")
            flush()
        
        transformer.train()
        if config.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if config.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            models_to_accumulate = [transformer]
            if config.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one, text_encoder_two])
                
            with accelerator.accumulate(models_to_accumulate):
                # Encode images
                if "latents" in batch:
                    latents = batch["latents"].to(weight_dtype)
                else:
                    latents = vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    latents = latents.to(weight_dtype)

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=config.logit_mean,
                    logit_std=config.logit_std,
                    mode_scale=config.mode_scale,
                )
                num_train_timesteps = int(noise_scheduler.config.num_train_timesteps)  # type: ignore
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
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)  # type: ignore

                # Add noise (Flow Matching)
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Get text embeddings
                if cached_embeddings and step in cached_embeddings:
                    prompt_embeds, pooled_embeds = cached_embeddings[step]
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                    pooled_embeds = pooled_embeds.to(accelerator.device)
                else:
                    prompt_embeds, pooled_embeds = encode_sd3_prompt(batch["prompts"], max_sequence_length=config.max_sequence_length or 256)

                # Predict
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]

                # Preconditioning
                if config.do_edm_style_training:
                    model_pred = model_pred * (-sigmas) + noisy_latents

                # Weighting
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)

                # Target
                if config.do_edm_style_training:
                    target = latents
                else:
                    target = noise - latents

                # Loss
                if config.use_prior_preservation:
                    # Chunk the model_pred and target into two parts, the first for the conditioning images and the second for the class images
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    weighting, weighting_prior = torch.chunk(weighting, 2, dim=0)

                    # Compute instance loss
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    ).mean()

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting_prior.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(target_prior.shape[0], -1),
                        1,
                    ).mean()

                    # Add the prior loss to the instance loss
                    loss = loss + config.prior_loss_weight * prior_loss
                else:
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(filter(lambda p: p.requires_grad, transformer.parameters()))
                    if config.train_text_encoder:
                        params_to_clip.extend(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
                        params_to_clip.extend(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if status_callback:
                    status_callback(global_step, max_train_steps, loss.item(), epoch)

                if config.save_every_n_steps and global_step % config.save_every_n_steps == 0:
                    cur_loss = float(loss.item())
                    if (not getattr(config, "save_best_only", False)) or (cur_loss < best_loss):
                        best_loss = min(best_loss, cur_loss)
                        save_lora_weights(
                            accelerator, transformer, config, global_step,
                            text_encoder_one if config.train_text_encoder else None,
                            text_encoder_two if config.train_text_encoder else None,
                            text_encoder_three if config.train_text_encoder else None,
                        )
                        manage_checkpoints(config.output_dir, config.checkpoints_total_limit, output_name=config.output_name)

                if config.sample_every_n_steps and global_step % config.sample_every_n_steps == 0:
                    if not config.train_text_encoder and config.cache_text_embeddings:
                        text_encoder_one.to(accelerator.device)
                        text_encoder_two.to(accelerator.device)
                        text_encoder_three.to(accelerator.device)
                        
                    pipeline = StableDiffusion3Pipeline(
                        transformer=accelerator.unwrap_model(transformer),
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        text_encoder_3=text_encoder_three,
                        tokenizer=tokenizer_one,
                        tokenizer_2=tokenizer_two,
                        tokenizer_3=tokenizer_three,
                        scheduler=noise_scheduler,
                    ).to(accelerator.device)
                    generate_samples(pipeline, config, global_step, epoch)
                    del pipeline
                    
                    if not config.train_text_encoder and config.cache_text_embeddings:
                        text_encoder_one.to("cpu")
                        text_encoder_two.to("cpu")
                        text_encoder_three.to("cpu")
                    flush()

        # Epoch end
        if config.sample_every_n_epochs and (epoch + 1) % config.sample_every_n_epochs == 0:
            if not config.train_text_encoder and config.cache_text_embeddings:
                text_encoder_one.to(accelerator.device)
                text_encoder_two.to(accelerator.device)
                text_encoder_three.to(accelerator.device)
                
            pipeline = StableDiffusion3Pipeline(
                transformer=accelerator.unwrap_model(transformer),
                vae=vae,
                text_encoder=accelerator.unwrap_model(text_encoder_one),
                text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                text_encoder_3=text_encoder_three,
                tokenizer=tokenizer_one,
                tokenizer_2=tokenizer_two,
                tokenizer_3=tokenizer_three,
                scheduler=noise_scheduler,
            ).to(accelerator.device)
            generate_samples(pipeline, config, global_step, epoch)
            del pipeline
            
            if not config.train_text_encoder and config.cache_text_embeddings:
                text_encoder_one.to("cpu")
                text_encoder_two.to("cpu")
                text_encoder_three.to("cpu")
            flush()

        if (epoch + 1) % config.save_every_n_epochs == 0:
            cur_loss = float(loss.item())
            if (not getattr(config, "save_best_only", False)) or (cur_loss < best_loss):
                best_loss = min(best_loss, cur_loss)
                save_lora_weights(
                    accelerator, transformer, config, global_step,
                    text_encoder_one if config.train_text_encoder else None,
                    text_encoder_two if config.train_text_encoder else None,
                    text_encoder_three if config.train_text_encoder else None,
                )
                manage_checkpoints(config.output_dir, config.checkpoints_total_limit, output_name=config.output_name)

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    # Final save
    save_lora_weights(accelerator, transformer, config, -1,
                      text_encoder_one if config.train_text_encoder else None,
                      text_encoder_two if config.train_text_encoder else None,
                      text_encoder_three if config.train_text_encoder else None)
    
    # Save model card
    save_model_card(
        config,
        config.output_dir,
        base_model=config.base_model_name,
        train_text_encoder=config.train_text_encoder,
    )
    
    if status_callback:
        status_callback(global_step, max_train_steps, 0, epoch, "Training completed.")

    # 10. Cleanup
    try:
        del transformer, vae, text_encoder_one, text_encoder_two, text_encoder_three, pipeline, noise_scheduler
        del optimizer, lr_scheduler, train_dataloader, accelerator
    except:
        pass
    flush()
    
    return True

