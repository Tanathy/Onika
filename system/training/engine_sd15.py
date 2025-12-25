# pyright: reportPrivateImportUsage=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportOptionalOperand=false
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
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import compute_snr, cast_training_params
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from safetensors.torch import save_file
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from .loss_utils import apply_noise_offset, fix_noise_scheduler_betas_for_zero_terminal_snr

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

class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.train_ids = None
        self.inserting_toks = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, inserting_toks: List[str]):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            self.inserting_toks = inserting_toks
            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            text_encoder.resize_token_embeddings(len(tokenizer))

            self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            std_token_embedding = text_encoder.get_input_embeddings().weight.data.std()

            text_encoder.get_input_embeddings().weight.data[self.train_ids] = (
                torch.randn(len(self.train_ids), text_encoder.config.hidden_size)
                .to(device=text_encoder.device, dtype=text_encoder.dtype)
                * std_token_embedding
            )
            self.embeddings_settings[f"original_embeddings_{idx}"] = (
                text_encoder.get_input_embeddings().weight.data.clone()
            )
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            inu = torch.ones((len(tokenizer),), dtype=torch.bool)
            inu[self.train_ids] = False
            self.embeddings_settings[f"index_no_updates_{idx}"] = inu
            idx += 1

    def save_embeddings(self, file_path: str):
        from safetensors.torch import save_file
        tensors = {}
        idx_to_name = {0: "clip_l", 1: "clip_g"}
        for idx, text_encoder in enumerate(self.text_encoders):
            new_token_embeddings = text_encoder.get_input_embeddings().weight.data[self.train_ids]
            tensors[idx_to_name.get(idx, f"text_encoder_{idx}")] = new_token_embeddings
        save_file(tensors, file_path)

    @torch.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            text_encoder.get_input_embeddings().weight.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device, dtype=text_encoder.dtype)
            )
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]
            index_updates = ~index_no_updates
            new_embeddings = text_encoder.get_input_embeddings().weight.data[index_updates]
            off_ratio = std_token_embedding / (new_embeddings.std() + 1e-8)
            new_embeddings = new_embeddings * (off_ratio**0.1)
            text_encoder.get_input_embeddings().weight.data[index_updates] = new_embeddings

def train_sd15(config: TrainingConfig, status_callback: Callable):
    # 1. Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    
    if config.seed is not None:
        set_seed(config.seed)

    # 2. Load Models
    status_callback(0, 0, 0, 0, f"Loading SD1.5 model from {config.base_model_name}...")
    
    # Auto-generate reg images if needed
    from .engine_utils import generate_class_images
    generate_class_images(config, accelerator, status_callback)
    
    load_kwargs = {
        "torch_dtype": torch.float32
    }
    if config.quantization_bit == "4":
        load_kwargs["load_in_4bit"] = True
    elif config.quantization_bit == "8":
        load_kwargs["load_in_8bit"] = True

    pipeline = StableDiffusionPipeline.from_single_file(
        config.base_model_name,
        **load_kwargs
    )
    
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    del pipeline
    flush()
    
    # 3. Prepare Models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
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
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    apply_optimizations(unet, config)
    
    # 4. Inject Network (LoRA, LoHa, LoKr, etc.)
    status_callback(0, 0, 0, 0, f"Injecting {config.network_type}...")
    
    embedding_handler = None
    if config.train_text_encoder_ti:
        # ... (TI logic remains same)
        token_abstraction_list = [t.strip() for t in config.token_abstraction.split(",") if t.strip()]
        token_abstraction_dict = {}
        token_idx = 0
        for i, token in enumerate(token_abstraction_list):
            token_abstraction_dict[token] = [f"<s{token_idx + i + j}>" for j in range(config.num_new_tokens_per_abstraction)]
            token_idx += config.num_new_tokens_per_abstraction - 1
        
        # Update prompts in config for the session
        for token_abs, token_replacement in token_abstraction_dict.items():
            config.instance_prompt = config.instance_prompt.replace(token_abs, "".join(token_replacement))
            if config.class_prompt:
                config.class_prompt = config.class_prompt.replace(token_abs, "".join(token_replacement))
        
        embedding_handler = TokenEmbeddingsHandler([text_encoder], [tokenizer])
        inserting_toks = []
        for new_tok in token_abstraction_dict.values():
            inserting_toks.extend(new_tok)
        embedding_handler.initialize_new_tokens(inserting_toks=inserting_toks)

    # LoRA setup
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
                # SD1.5 UNet structure: down_blocks, up_blocks, mid_block
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
        text_encoder = inject_network(text_encoder, config, te_target_modules, is_text_encoder=True)

    # Cast trainable params to float32
    if accelerator.mixed_precision == "fp16":
        models = [unet]
        if config.train_text_encoder or config.train_text_encoder_ti:
            models.extend([text_encoder])
        cast_training_params(models, dtype=torch.float32)

    # 5. Optimizer
    params_to_optimize = [{"params": list(filter(lambda p: p.requires_grad, unet.parameters())), "lr": config.learning_rate}]
    
    if config.train_text_encoder:
        params_to_optimize.append({
            "params": list(filter(lambda p: p.requires_grad, text_encoder.parameters())),
            "lr": config.text_encoder_lr if config.text_encoder_lr else config.learning_rate
        })
    elif config.train_text_encoder_ti:
        text_lora_parameters = []
        for name, param in text_encoder.named_parameters():
            if "token_embedding" in name:
                param.requires_grad = True
                text_lora_parameters.append(param)
            else:
                param.requires_grad = False
        params_to_optimize.append({
            "params": text_lora_parameters,
            "lr": config.text_encoder_lr if config.text_encoder_lr else config.learning_rate
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
        collate_fn=collate_fn_general,
        num_workers=config.dataloader_num_workers,
        persistent_workers=persistent_workers if int(getattr(config, "dataloader_num_workers", 0) or 0) > 0 else False,
    )
    
    # 7. Prepare with Accelerator
    if config.train_text_encoder or config.train_text_encoder_ti:
        unet, text_encoder, optimizer, dataloader = accelerator.prepare(unet, text_encoder, optimizer, dataloader)
    else:
        unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    # 8. Training Loop
    num_update_steps_per_epoch = len(dataloader) // config.gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, int(num_update_steps_per_epoch))

    max_train_steps = int(config.max_train_steps) if config.max_train_steps else (config.max_train_epochs * num_update_steps_per_epoch)
    max_train_steps = max(1, int(max_train_steps))
    planned_epochs = int((max_train_steps + num_update_steps_per_epoch - 1) // num_update_steps_per_epoch)
    planned_epochs = max(1, planned_epochs)

    if config.zero_terminal_snr:
        fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
    
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
            save_lora_weights(accelerator, unet, config, global_step, 
                              text_encoder if config.train_text_encoder else None)

    def load_model_hook(models, input_dir):
        pass

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
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
            inputs = tokenizer(batch["prompts"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            encoder_hidden_states = text_encoder(inputs.input_ids.to(accelerator.device))[0]
            cached_embeddings[i] = encoder_hidden_states.cpu()
        
        if not config.train_text_encoder and not config.train_text_encoder_ti:
            text_encoder.to("cpu")
            flush()

    status_callback(global_step, max_train_steps, 0, first_epoch, "Starting training loop...")

    def per_element_loss(pred, tgt):
        lt = (getattr(config, "loss_type", None) or "l2").strip().lower()
        if lt in {"l1", "mae"}:
            return F.l1_loss(pred.float(), tgt.float(), reduction="none")
        if lt == "huber":
            return F.huber_loss(pred.float(), tgt.float(), reduction="none", delta=float(getattr(config, "huber_c", 0.1) or 0.1))
        if lt in {"smooth_l1", "smoothl1"}:
            return F.smooth_l1_loss(pred.float(), tgt.float(), reduction="none")
        return F.mse_loss(pred.float(), tgt.float(), reduction="none")
    
    num_train_epochs_text_encoder = int(config.train_text_encoder_frac * planned_epochs)
    if config.train_text_encoder_ti:
        num_train_epochs_text_encoder = int(config.train_text_encoder_ti_frac * planned_epochs)

    stop_training = False
    best_loss = float("inf")
    for epoch in range(first_epoch, planned_epochs):
        # Handle TE/TI freezing
        if config.train_text_encoder or config.train_text_encoder_ti:
            if epoch >= num_train_epochs_text_encoder:
                # Stop optimizing TE/TI
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = 0.0
            else:
                text_encoder.train()

        unet.train()
        for step, batch in enumerate(dataloader):
            models_to_accumulate = [unet]
            if config.train_text_encoder or config.train_text_encoder_ti:
                models_to_accumulate.append(text_encoder)
                
            with accelerator.accumulate(models_to_accumulate):
                # Latents
                if "latents" in batch:
                    latents = batch["latents"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)
                
                # Noise
                noise = torch.randn_like(latents)
                if config.noise_offset_strength > 0 or config.noise_offset_random_strength > 0:
                    noise = apply_noise_offset(
                        noise,
                        noise_offset_type=getattr(config, "noise_offset_type", "original"),
                        noise_offset=float(config.noise_offset_strength or 0.0),
                        noise_offset_random_strength=float(config.noise_offset_random_strength or 0.0),
                        adaptive_noise_scale=float(config.adaptive_noise_scale or 0.0) if (config.adaptive_noise_scale or 0.0) > 0 else None,
                        latents=latents,
                    )
                
                bsz = latents.shape[0]

                num_train_timesteps = int(noise_scheduler.config.num_train_timesteps)
                min_ts = 0 if config.min_timestep is None else int(config.min_timestep)
                max_ts = (num_train_timesteps - 1) if config.max_timestep is None else int(config.max_timestep)
                min_ts = max(0, min_ts)
                max_ts = min(num_train_timesteps - 1, max_ts)
                if min_ts > max_ts:
                    min_ts, max_ts = 0, num_train_timesteps - 1

                timesteps = torch.randint(min_ts, max_ts + 1, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Prompts
                if cached_embeddings and step in cached_embeddings:
                    encoder_hidden_states = cached_embeddings[step].to(accelerator.device)
                else:
                    inputs = tokenizer(batch["prompts"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    encoder_hidden_states = text_encoder(inputs.input_ids.to(accelerator.device))[0]

                # Predict
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                # Min-SNR Gamma
                snr_gamma = config.snr_gamma if config.snr_gamma is not None else config.min_snr_gamma
                if snr_gamma is not None and snr_gamma > 0:
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

                if embedding_handler:
                    embedding_handler.retract_embeddings()
            
            if accelerator.sync_gradients:
                global_step += 1
                status_callback(global_step, max_train_steps, loss.item(), epoch, None)

                # Save every n steps
                if config.save_every_n_steps and global_step % config.save_every_n_steps == 0:
                    cur_loss = float(loss.item())
                    if (not getattr(config, "save_best_only", False)) or (cur_loss < best_loss):
                        best_loss = min(best_loss, cur_loss)
                        save_lora_weights(
                            accelerator, unet, config, global_step,
                            text_encoder if config.train_text_encoder else None,
                        )
                        manage_checkpoints(config.output_dir, config.checkpoints_total_limit, output_name=config.output_name)

                # Sample generation
                if config.sample_every_n_steps and global_step % config.sample_every_n_steps == 0:
                    if not config.train_text_encoder and not config.train_text_encoder_ti and config.cache_text_embeddings:
                        text_encoder.to(accelerator.device)
                        
                    pipeline = StableDiffusionPipeline(
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        unet=accelerator.unwrap_model(unet),
                        scheduler=noise_scheduler,
                        safety_checker=None,
                        feature_extractor=None,
                    )
                    generate_samples(pipeline, config, global_step, epoch)
                    del pipeline
                    
                    if not config.train_text_encoder and not config.train_text_encoder_ti and config.cache_text_embeddings:
                        text_encoder.to("cpu")
                    flush()

                if global_step >= max_train_steps:
                    stop_training = True
                    break

        if stop_training:
            break

        # Epoch end
        if config.sample_every_n_epochs and (epoch + 1) % config.sample_every_n_epochs == 0:
            if not config.train_text_encoder and not config.train_text_encoder_ti and config.cache_text_embeddings:
                text_encoder.to(accelerator.device)
                
            pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                unet=accelerator.unwrap_model(unet),
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
            )
            generate_samples(pipeline, config, global_step, epoch)
            del pipeline
            
            if not config.train_text_encoder and not config.train_text_encoder_ti and config.cache_text_embeddings:
                text_encoder.to("cpu")
            flush()

        if (epoch + 1) % config.save_every_n_epochs == 0:
            cur_loss = float(loss.item())
            if (not getattr(config, "save_best_only", False)) or (cur_loss < best_loss):
                best_loss = min(best_loss, cur_loss)
                save_lora_weights(
                    accelerator, unet, config, global_step,
                    text_encoder if config.train_text_encoder else None,
                )
                manage_checkpoints(config.output_dir, config.checkpoints_total_limit, output_name=config.output_name)

    # 9. Save Final
    status_callback(global_step, max_train_steps, 0, 0, "Saving Final LoRA...")
    save_lora_weights(accelerator, unet, config, -1, 
                      text_encoder if config.train_text_encoder else None)
    
    if embedding_handler:
        emb_path = Path(config.output_dir) / f"{config.output_name}_emb.safetensors"
        embedding_handler.save_embeddings(str(emb_path))
        print(f"Saved TI embeddings to {emb_path}")
    
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
        del unet, vae, text_encoder, pipeline, noise_scheduler
        del optimizer, lr_scheduler, train_dataloader, accelerator
    except:
        pass
    flush()

