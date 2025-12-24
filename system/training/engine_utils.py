import os
import torch
import random
import shutil
import gc
import numpy as np
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from system.log import info, error
from system.coordinator_settings import SETTINGS


def _make_divisible_by_8(size: int) -> int:
    """Round down to nearest multiple of 8"""
    return (size // 8) * 8


def _center_crop_to_square_div8(image: Image.Image, target_size: int) -> tuple:
    """
    Center crop image to square, ensuring final size is divisible by 8.
    Returns (cropped_image, crop_box, final_size)
    """
    w, h = image.size
    
    # Determine the square size (minimum of width/height)
    square_size = min(w, h)
    
    # If target_size is specified and smaller, use that
    if target_size and target_size < square_size:
        square_size = target_size
    
    # Make divisible by 8 (round down slightly if needed)
    final_size = _make_divisible_by_8(square_size)
    if final_size == 0:
        final_size = 8  # Minimum size
    
    # Calculate crop box for center crop
    left = (w - final_size) // 2
    top = (h - final_size) // 2
    right = left + final_size
    bottom = top + final_size
    
    crop_box = (left, top, right, bottom)
    cropped = image.crop(crop_box)
    
    return cropped, crop_box, final_size


def get_latents_cache_dir(config: Any) -> Path:
    # Keep it simple + predictable: always under project cache.
    # Partition by model_type + resolution to avoid accidental cross-architecture reuse.
    model_type = getattr(config, "model_type", "unknown") or "unknown"
    resolution = getattr(config, "resolution", "unknown")
    return Path("project") / "cache" / "latents" / str(model_type) / str(resolution)


def _latent_cache_relpath(dataset_root: Path, image_path: Path) -> str:
    try:
        return str(image_path.relative_to(dataset_root)).replace("\\", "/")
    except Exception:
        return str(image_path).replace("\\", "/")


def _latent_cache_key(dataset_root: Path, image_path: Path) -> str:
    # Stable key across runs as long as dataset folder structure stays the same.
    rel = _latent_cache_relpath(dataset_root, image_path)
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]


def _latent_cache_paths(cache_dir: Path, kind: str, dataset_root: Path, image_path: Path) -> List[Path]:
    """
    Returns candidate cache paths in priority order.

    New format (preferred):
      <cache_dir>/<kind>/<hash>.pt

    Legacy format (backward-compat):
      <cache_dir>/<kind>__<stem>__<hash>.pt
    """
    key = _latent_cache_key(dataset_root, image_path)
    new_path = cache_dir / kind / f"{key}.pt"
    legacy_path = cache_dir / f"{kind}__{image_path.stem}__{key}.pt"
    return [new_path, legacy_path]


def _find_latent_cache_path(cache_dir: Path, kind: str, dataset_root: Path, image_path: Path) -> Path:
    for p in _latent_cache_paths(cache_dir, kind, dataset_root, image_path):
        if p.exists():
            return p
    # Default to new path for error messages / future writes.
    return _latent_cache_paths(cache_dir, kind, dataset_root, image_path)[0]

def collate_fn_general(examples):
    prompts = [example["instance_prompt"] for example in examples]

    if "latents" in examples[0]:
        latents = torch.stack([example["latents"] for example in examples])
        batch: Dict[str, Any] = {"latents": latents, "prompts": prompts}

        if "class_latents" in examples[0]:
            class_latents = torch.stack([example["class_latents"] for example in examples])
            batch["latents"] = torch.cat([latents, class_latents], dim=0)
            batch["prompts"] = prompts + [example["class_prompt"] for example in examples]

        return batch

    pixel_values = torch.stack([example["instance_images"] for example in examples])
    if "class_images" in examples[0]:
        class_pixel_values = torch.stack([example["class_images"] for example in examples])
        pixel_values = torch.cat([pixel_values, class_pixel_values], dim=0)
        prompts = prompts + [example["class_prompt"] for example in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}

def collate_fn_sdxl(examples):
    prompts = [example["instance_prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]
    
    batch = {
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts
    }

    if "latents" in examples[0]:
        latents = torch.stack([example["latents"] for example in examples])
        if "class_latents" in examples[0]:
            class_latents = torch.stack([example["class_latents"] for example in examples])
            latents = torch.cat([latents, class_latents], dim=0)

            class_original_sizes = [example["class_original_size"] for example in examples]
            class_crop_top_lefts = [example["class_crop_top_left"] for example in examples]

            batch["prompts"] = prompts + [example["class_prompt"] for example in examples]
            batch["original_sizes"] = original_sizes + class_original_sizes
            batch["crop_top_lefts"] = crop_top_lefts + class_crop_top_lefts
        batch["latents"] = latents
    else:
        pixel_values = torch.stack([example["instance_images"] for example in examples])
        if "class_images" in examples[0]:
            class_pixel_values = torch.stack([example["class_images"] for example in examples])
            pixel_values = torch.cat([pixel_values, class_pixel_values], dim=0)
            batch["prompts"] = prompts + [example["class_prompt"] for example in examples]
            batch["original_sizes"] = original_sizes + original_sizes
            batch["crop_top_lefts"] = crop_top_lefts + crop_top_lefts
        batch["pixel_values"] = pixel_values
        
    return batch

class OnikaDataset(Dataset):
    def __init__(
        self,
        config: Any, # TrainingConfig
    ):
        self.config = config
        self.size = _make_divisible_by_8(config.resolution)  # Ensure resolution is divisible by 8
        self.instance_data_root = Path(config.dataset_path)
        
        # Debug mode setup
        self.debug_mode = SETTINGS.get("debug_mode", False)
        self.debug_dir: Optional[Path] = None
        self._debug_save_count = 0
        if self.debug_mode:
            self.debug_dir = Path("project/debug")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            # Clear old debug images
            for old_file in self.debug_dir.glob("*.png"):
                old_file.unlink()
            info(f"Debug mode enabled. Saving processed images to {self.debug_dir}")
        
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance images root '{config.dataset_path}' doesn't exist.")

        self.instance_images_paths = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for path in self.instance_data_root.iterdir():
            if path.suffix.lower() in valid_exts:
                self.instance_images_paths.append(path)
        
        self.num_instance_images = len(self.instance_images_paths)
        self._length = self.num_instance_images

        # Latent caching (disk)
        self.cache_dir: Optional[Path] = None
        self.use_cached_latents = False

        # Augmentation transforms (applied AFTER crop, before tensor conversion)
        # NOTE: Latent caching is incompatible with realtime image augmentation.
        self.aug_transforms: List[Any] = []
        if getattr(config, "cache_latents", False) or getattr(config, "cache_latents_to_disk", False):
            # Leave aug_transforms empty on purpose.
            pass
        else:
            if config.flip_aug_probability > 0:
                self.aug_transforms.append(transforms.RandomHorizontalFlip(p=config.flip_aug_probability))
            
            if config.color_aug_strength > 0:
                self.aug_transforms.append(transforms.ColorJitter(
                    brightness=config.color_aug_strength * 0.2,
                    contrast=config.color_aug_strength * 0.2,
                    saturation=config.color_aug_strength * 0.2,
                    hue=config.color_aug_strength * 0.1
                ))
        
        # Final transforms (to tensor)
        self.final_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Prior Preservation
        self.class_data_root = Path(config.reg_data_dir) if config.reg_data_dir else None
        self.class_prompt = config.class_prompt
        if self.class_data_root and self.class_data_root.exists():
            self.class_images_paths = []
            for path in self.class_data_root.iterdir():
                if path.suffix.lower() in valid_exts:
                    self.class_images_paths.append(path)
            self.num_class_images = len(self.class_images_paths)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_images_paths = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        # Instance Image
        instance_path = self.instance_images_paths[index % self.num_instance_images]
        
        # Check for cached latents
        if self.use_cached_latents and self.cache_dir:
            latent_path = _find_latent_cache_path(self.cache_dir, "instance", self.instance_data_root, instance_path)
            if latent_path.exists():
                try:
                    cached_data = torch.load(latent_path, map_location="cpu")
                    example["latents"] = cached_data["latents"]
                    example["original_size"] = cached_data["original_size"]
                    example["crop_top_left"] = cached_data["crop_top_left"]
                    
                    # Caption logic (still needed)
                    caption_path = instance_path.with_suffix(self.config.caption_extension or ".txt")
                    caption = ""
                    if caption_path.exists():
                        with open(caption_path, "r", encoding="utf-8") as f:
                            caption = f.read().strip()
                    else:
                        caption = self.config.instance_prompt or ""

                    if self.config.caption_dropout_rate > 0 and random.random() < self.config.caption_dropout_rate:
                        caption = ""
                    
                    if self.config.shuffle_caption and caption:
                        tokens = caption.split(",")
                        if len(tokens) > 1:
                            keep = tokens[:self.config.keep_tokens]
                            rest = tokens[self.config.keep_tokens:]
                            random.shuffle(rest)
                            caption = ",".join(keep + rest)

                    example["instance_prompt"] = caption

                    # Prior preservation: if we have class images, we MUST provide them too.
                    if self.class_images_paths:
                        class_path = self.class_images_paths[index % self.num_class_images]
                        if not self.class_data_root:
                            raise RuntimeError("class_data_root is not set but class_images_paths is non-empty")

                        class_latent_path = _find_latent_cache_path(self.cache_dir, "class", self.class_data_root, class_path)
                        if not class_latent_path.exists():
                            raise FileNotFoundError(
                                f"Missing cached class latent: {class_latent_path}. "
                                "Delete the latents cache folder and re-run caching."
                            )

                        class_cached = torch.load(class_latent_path, map_location="cpu")
                        example["class_latents"] = class_cached["latents"]
                        example["class_original_size"] = class_cached["original_size"]
                        example["class_crop_top_left"] = class_cached["crop_top_left"]
                        example["class_prompt"] = self.class_prompt

                    return example
                except Exception as e:
                    raise
            else:
                raise FileNotFoundError(
                    f"Missing cached instance latent: {latent_path}. "
                    "Delete the latents cache folder and re-run caching."
                )
        
        instance_image = Image.open(instance_path)
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        original_size = instance_image.size  # (w, h) before any processing
        
        # ALWAYS center crop to square with size divisible by 8
        # This applies regardless of bucketing settings
        cropped_image, crop_box, final_size = _center_crop_to_square_div8(instance_image, self.size)
        
        # Resize to target resolution if needed (also divisible by 8)
        if cropped_image.size[0] != self.size:
            cropped_image = cropped_image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Save debug image (cropped, before augmentation)
        if self.debug_mode and self.debug_dir and self._debug_save_count < 50:  # Limit debug saves
            debug_name = f"crop_{self._debug_save_count:03d}_{instance_path.stem}.png"
            cropped_image.save(self.debug_dir / debug_name)
        
        # Apply augmentations (flip, color jitter, etc.)
        augmented_image = cropped_image
        for aug in self.aug_transforms:
            augmented_image = aug(augmented_image)
        
        # Save debug image (after augmentation)
        if self.debug_mode and self.debug_dir and self._debug_save_count < 50:
            debug_name = f"aug_{self._debug_save_count:03d}_{instance_path.stem}.png"
            # Convert back to PIL if it's a tensor (shouldn't be at this point)
            if isinstance(augmented_image, Image.Image):
                augmented_image.save(self.debug_dir / debug_name)
            self._debug_save_count += 1
        
        # For SDXL micro-conditioning
        example["original_size"] = (original_size[1], original_size[0])  # (h, w) format
        example["crop_top_left"] = (crop_box[1], crop_box[0])  # (top, left) format
        
        # Final transforms (to tensor)
        example["instance_images"] = self.final_transforms(augmented_image)
        
        # Caption with Dropout
        caption_path = instance_path.with_suffix(self.config.caption_extension or ".txt")
        caption = ""
        if caption_path.exists():
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            caption = self.config.instance_prompt or ""

        # Caption Dropout
        if self.config.caption_dropout_rate > 0 and random.random() < self.config.caption_dropout_rate:
            caption = ""
        
        # Shuffle Caption (Simplified)
        if self.config.shuffle_caption and caption:
            tokens = caption.split(",")
            if len(tokens) > 1:
                # Keep first N tokens if specified
                keep = tokens[:self.config.keep_tokens]
                rest = tokens[self.config.keep_tokens:]
                random.shuffle(rest)
                caption = ",".join(keep + rest)

        example["instance_prompt"] = caption

        # Class Image (Prior Preservation)
        if self.class_images_paths:
            class_path = self.class_images_paths[index % self.num_class_images]
            class_image = Image.open(class_path)
            class_image = exif_transpose(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            
            # Apply same cropping logic to class images
            class_cropped, _, _ = _center_crop_to_square_div8(class_image, self.size)
            if class_cropped.size[0] != self.size:
                class_cropped = class_cropped.resize((self.size, self.size), Image.Resampling.LANCZOS)
            
            # Apply augmentations
            for aug in self.aug_transforms:
                class_cropped = aug(class_cropped)
            
            example["class_images"] = self.final_transforms(class_cropped)
            example["class_prompt"] = self.class_prompt

        return example

def get_optimizer(config, params):
    optimizer_name = config.optimizer_type.lower()
    
    # Parse optimizer args
    opt_kwargs = {}
    if config.optimizer_args:
        def _split_args(s: str):
            parts = []
            buf = []
            depth = 0
            for ch in s:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}" and depth > 0:
                    depth -= 1
                if ch == "," and depth == 0:
                    part = "".join(buf).strip()
                    if part:
                        parts.append(part)
                    buf = []
                    continue
                buf.append(ch)
            tail = "".join(buf).strip()
            if tail:
                parts.append(tail)
            return parts

        if isinstance(config.optimizer_args, str):
            args = _split_args(config.optimizer_args)
        else:
            args = list(config.optimizer_args)

        def _parse_value(k: str, v: str):
            raw = v.strip()
            if raw.lower() in ("true", "false"):
                return raw.lower() == "true"
            if k.strip().lower() == "betas":
                cleaned = raw.strip().strip("()[]{}").replace(" ", "")
                for sep in (",", ":", ";", "|"):
                    if sep in cleaned:
                        a, b = cleaned.split(sep, 1)
                        return (float(a), float(b))
                # Allow single number -> interpret as beta1 and use default beta2
                return (float(cleaned), 0.999)
            try:
                if any(c in raw for c in (".", "e", "E")):
                    return float(raw)
                return int(raw)
            except Exception:
                return raw

        for arg in args:
            if not isinstance(arg, str):
                continue
            if "=" not in arg:
                continue
            k, v = arg.split("=", 1)
            key = k.strip()
            opt_kwargs[key] = _parse_value(key, v)

    # Normalize betas for optimizers that expect a 2-tuple
    beta1 = opt_kwargs.pop("beta1", None)
    beta2 = opt_kwargs.pop("beta2", None)
    if "betas" in opt_kwargs and (beta1 is not None or beta2 is not None):
        # Prefer explicit betas if provided
        beta1 = None
        beta2 = None

    if "betas" in opt_kwargs:
        betas = opt_kwargs["betas"]
        if isinstance(betas, (list, tuple)) and len(betas) == 2:
            opt_kwargs["betas"] = (float(betas[0]), float(betas[1]))
        else:
            raise ValueError(
                f"Invalid optimizer arg 'betas': expected two values, got {betas!r}. "
                "Use e.g. betas=(0.9,0.999)"
            )
    elif beta1 is not None or beta2 is not None:
        opt_kwargs["betas"] = (float(beta1) if beta1 is not None else 0.9, float(beta2) if beta2 is not None else 0.999)

    try:
        if optimizer_name == "adamw8bit":
            import bitsandbytes as bnb # type: ignore
            return bnb.optim.AdamW8bit(params, lr=config.learning_rate, **opt_kwargs) # type: ignore
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=config.learning_rate, **opt_kwargs)
        elif optimizer_name == "prodigy":
            import prodigyopt
            return prodigyopt.Prodigy(params, lr=config.learning_rate, **opt_kwargs)
        elif optimizer_name == "lion":
            from lion_pytorch import Lion
            return Lion(params, lr=config.learning_rate, **opt_kwargs)
        elif optimizer_name == "lion8bit":
            import bitsandbytes as bnb # type: ignore
            return bnb.optim.Lion8bit(params, lr=config.learning_rate, **opt_kwargs) # type: ignore
        else:
            # Fallback to AdamW
            return torch.optim.AdamW(params, lr=config.learning_rate, **opt_kwargs)
    except Exception as e:
        from system.log import error
        error(
            f"Optimizer init failed ({config.optimizer_type}, args={config.optimizer_args}). "
            f"Parsed kwargs={opt_kwargs}. Error: {e}"
        )
        raise

def apply_optimizations(model, config):
    if config.attention_backend == "xformers" or config.xformers:
        try:
            model.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Could not enable xformers: {e}")
    elif config.attention_backend == "sdpa":
        # SDPA is usually default in PT 2.0+
        pass
    
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def generate_samples(pipeline, config: Any, step: int, epoch: int):
    pipeline.set_progress_bar_config(disable=True)
    output_dir = Path(config.output_dir) / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = config.sample_prompts
    prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]
    if not prompt_list:
        return
    
    generator = torch.Generator(device=pipeline.device)
    if config.sample_seed is not None and config.sample_seed != -1:
        generator.manual_seed(config.sample_seed)
    
    for i, prompt in enumerate(prompt_list):
        for j in range(config.num_validation_images):
            image = pipeline(
                prompt,
                negative_prompt=config.sample_negative_prompt,
                num_inference_steps=config.sample_num_inference_steps,
                guidance_scale=config.sample_guidance_scale,
                generator=generator
            ).images[0]
            
            filename = f"sample_e{epoch}_s{step}_{i}_{j}.png"
            image.save(output_dir / filename)

def inject_network(model, config, target_modules, is_text_encoder=False):
    from peft import LoraConfig, LoHaConfig, LoKrConfig
    
    network_type = config.network_type.lower()
    
    # Common arguments
    kwargs = {
        "r": config.network_dim,
        "target_modules": target_modules,
    }
    
    # Alpha handling
    # For LoRA, alpha is usually rank or a fixed value.
    # For LoHa/LoKr, it's also used.
    network_alpha = config.network_alpha
    
    if network_type == "lora":
        peft_config = LoraConfig(
            lora_alpha=network_alpha,
            lora_dropout=config.network_dropout,
            init_lora_weights="gaussian",
            **kwargs
        )
    elif network_type == "loha":
        peft_config = LoHaConfig(
            alpha=network_alpha,
            rank_dropout=config.rank_dropout,
            module_dropout=config.module_dropout,
            init_weights=True,
            **kwargs
        )
    elif network_type == "lokr":
        peft_config = LoKrConfig(
            alpha=network_alpha,
            rank_dropout=config.rank_dropout,
            module_dropout=config.module_dropout,
            init_weights=True,
            **kwargs
        )
    elif network_type == "lycoris":
        # Default to LoHa for LyCORIS if not specified, or use algo
        algo = (config.algo or "loha").lower()
        if algo == "loha":
            peft_config = LoHaConfig(
                alpha=network_alpha,
                rank_dropout=config.rank_dropout,
                module_dropout=config.module_dropout,
                init_weights=True,
                **kwargs
            )
        elif algo == "lokr":
            peft_config = LoKrConfig(
                alpha=network_alpha,
                rank_dropout=config.rank_dropout,
                module_dropout=config.module_dropout,
                init_weights=True,
                **kwargs
            )
        else:
            # Fallback to LoRA
            peft_config = LoraConfig(
                lora_alpha=network_alpha,
                lora_dropout=config.network_dropout,
                init_lora_weights="gaussian",
                **kwargs
            )
    else:
        # Default to LoRA
        peft_config = LoraConfig(
            lora_alpha=network_alpha,
            lora_dropout=config.network_dropout,
            init_lora_weights="gaussian",
            **kwargs
        )
        
    model.add_adapter(peft_config)
    return model

def save_lora_weights(accelerator, model, config, step, text_encoder=None, text_encoder_2=None, text_encoder_3=None):
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file
    import json
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename
    if step == -1:
        filename = f"{config.output_name}.safetensors"
    else:
        filename = f"{config.output_name}_s{step}.safetensors"
    
    save_path = output_dir / filename
    
    # Get model state dict
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = get_peft_model_state_dict(unwrapped_model)
    
    # Add text encoder weights if provided
    if text_encoder is not None:
        te_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(text_encoder))
        # Prefix keys to avoid collision
        te_state_dict = {f"text_encoder.{k}": v for k, v in te_state_dict.items()}
        state_dict.update(te_state_dict)
        
    if text_encoder_2 is not None:
        te2_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(text_encoder_2))
        te2_state_dict = {f"text_encoder_2.{k}": v for k, v in te2_state_dict.items()}
        state_dict.update(te2_state_dict)

    if text_encoder_3 is not None:
        te3_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(text_encoder_3))
        te3_state_dict = {f"text_encoder_3.{k}": v for k, v in te3_state_dict.items()}
        state_dict.update(te3_state_dict)
    
    # Metadata from PEFT configs (following diffusers example)
    peft_metadata = {}
    modules_to_save = {"transformer" if config.model_type in ["flux1", "flux2", "sd3", "sd3.5"] else "unet": unwrapped_model}
    if text_encoder: modules_to_save["text_encoder"] = accelerator.unwrap_model(text_encoder)
    if text_encoder_2: modules_to_save["text_encoder_2"] = accelerator.unwrap_model(text_encoder_2)
    if text_encoder_3: modules_to_save["text_encoder_3"] = accelerator.unwrap_model(text_encoder_3)
    
    for module_name, module in modules_to_save.items():
        if hasattr(module, "peft_config") and "default" in module.peft_config:
            # Handle non-serializable objects like sets in the config
            config_dict = module.peft_config["default"].to_dict()
            def _serialize(obj):
                if isinstance(obj, set):
                    return list(obj)
                return str(obj)
            peft_metadata[f"{module_name}_lora_adapter_metadata"] = json.dumps(config_dict, default=_serialize)

    # Convert to Kohya format for compatibility
    state_dict = convert_to_kohya_format(state_dict, config.model_type, config.network_alpha)
    
    # Convert to save precision
    save_dtype = torch.float32
    if config.save_precision == "float16":
        save_dtype = torch.float16
    elif config.save_precision == "bf16":
        save_dtype = torch.bfloat16
    
    state_dict = {k: v.to(save_dtype) for k, v in state_dict.items()}
    
    # Global Metadata
    metadata = {
        "ss_output_name": config.output_name,
        "ss_base_model_name": config.base_model_name,
        "ss_network_dim": str(config.network_dim),
        "ss_network_alpha": str(config.network_alpha),
        "ss_network_module": config.network_module,
        "ss_learning_rate": str(config.learning_rate),
        "ss_num_epochs": str(config.max_train_epochs),
        "ss_resolution": str(config.resolution),
        "ss_training_comment": config.training_comment or "",
        "ss_metadata_title": config.metadata_title or "",
        "ss_metadata_author": config.metadata_author or "",
        "ss_metadata_description": config.metadata_description or "",
        "ss_metadata_license": config.metadata_license or "",
        "ss_metadata_tags": config.metadata_tags or "",
        "ss_model_type": "flux" if config.model_type in ["flux1", "flux2"] else config.model_type,
        "ss_v2": "False",
        "ss_sd_model_name": config.base_model_name,
        
        # Extended Metadata for A1111/ComfyUI Compatibility
        "ss_clip_skip": str(config.clip_skip) if config.clip_skip is not None else "None",
        "ss_max_train_steps": str(config.max_train_steps) if config.max_train_steps else "None",
        "ss_lr_scheduler": config.lr_scheduler,
        "ss_lr_warmup_steps": str(config.lr_warmup_steps),
        "ss_text_encoder_lr": str(config.text_encoder_lr) if config.text_encoder_lr else "None",
        "ss_unet_lr": str(config.unet_lr) if config.unet_lr else "None",
        "ss_optimizer": f"{config.optimizer_type} (args={config.optimizer_args})",
        "ss_cache_latents": str(config.cache_latents),
        "ss_gradient_checkpointing": str(config.gradient_checkpointing),
        "ss_gradient_accumulation_steps": str(config.gradient_accumulation_steps),
        "ss_mixed_precision": config.mixed_precision,
        "ss_seed": str(config.seed) if config.seed is not None else "None",
        "ss_lowram": str(config.enable_aggressive_memory_saving),
        "ss_noise_offset": str(config.noise_offset_strength) if config.noise_offset_strength > 0 else "None",
        "ss_adaptive_noise_scale": str(config.adaptive_noise_scale) if config.adaptive_noise_scale > 0 else "None",
        "ss_multires_noise_iterations": str(config.multires_noise_iterations) if config.multires_noise_iterations > 0 else "None",
        "ss_multires_noise_discount": str(config.multires_noise_discount) if config.multires_noise_discount > 0 else "None",
        "ss_min_snr_gamma": str(config.min_snr_gamma) if config.min_snr_gamma else "None",
        "ss_batch_size_per_device": str(config.batch_size),
        "ss_bucket_no_upscale": str(config.bucket_no_upscale),
        "ss_enable_bucket": str(config.enable_bucket),
        "ss_min_bucket_reso": str(config.min_bucket_reso),
        "ss_max_bucket_reso": str(config.max_bucket_reso),
        "ss_keep_tokens": str(config.keep_tokens),
        "ss_shuffle_caption": str(config.shuffle_caption),
        "ss_caption_dropout_rate": str(config.caption_dropout_rate),
        "ss_caption_dropout_every_n_epochs": str(config.caption_dropout_every_n_epochs),
        "ss_color_aug": str(config.color_aug_strength > 0),
        "ss_flip_aug": str(config.flip_aug_probability > 0),
        "ss_face_crop_aug_range": "None", # Not implemented yet
        "ss_random_crop": str(config.random_crop_scale < 1.0),
        "ss_full_fp16": str(config.full_fp16),
        "ss_zero_terminal_snr": str(config.zero_terminal_snr) if hasattr(config, "zero_terminal_snr") else "False",
        "ss_max_token_length": str(config.max_token_length) if config.max_token_length else "None",
        "ss_prior_loss_weight": str(config.prior_loss_weight),
        "ss_scale_weight_norms": str(config.scale_weight_norms) if config.scale_weight_norms else "None",
        "ss_network_dropout": str(config.network_dropout),
        "ss_dataset_dirs": str(config.dataset_path),
        "ss_tag_frequency": "{}", # Placeholder
        "ss_bucket_info": "{}", # Placeholder
    }
    metadata.update(peft_metadata)
    
    save_file(state_dict, save_path, metadata=metadata)
    print(f"Saved LoRA weights to {save_path}")

def save_model_card(
    config: Any,
    repo_folder: str,
    images: Optional[List[Image.Image]] = None,
    base_model: Optional[str] = None,
    train_text_encoder: bool = False,
    instance_prompt: Optional[str] = None,
    validation_prompt: Optional[str] = None,
):
    from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
    
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image_name = f"image_{i}.png"
            image.save(os.path.join(repo_folder, image_name))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": image_name}}
            )

    model_description = f"""
# {config.model_type.upper()} LoRA DreamBooth - {config.output_name}

<Gallery />

## Model description

These are {config.output_name} LoRA adaption weights for {base_model or config.base_model_name}.

The weights were trained using [Onika](https://github.com/Onika-AI/Onika).

- **Architecture**: {config.model_type}
- **Network Type**: {config.network_type}
- **Rank**: {config.network_dim}
- **Alpha**: {config.network_alpha}
- **Text Encoder Training**: {train_text_encoder}

## Trigger words

You should use `{instance_prompt or config.instance_prompt}` to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.
"""
    
    model_card = load_or_create_model_card(
        repo_id_or_path=config.output_name,
        from_training=True,
        base_model=base_model or config.base_model_name,
        prompt=instance_prompt or config.instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "template:sd-lora",
        config.model_type,
    ]
    
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))

def manage_checkpoints(output_dir: str, checkpoints_total_limit: Optional[int]):
    if checkpoints_total_limit is None or checkpoints_total_limit <= 0:
        return

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # we delete the oldest checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_delete = len(checkpoints) - checkpoints_total_limit + 1
        for i in range(num_to_delete):
            shutil.rmtree(os.path.join(output_dir, checkpoints[i]))

def convert_to_kohya_format(state_dict, model_type, network_alpha=None):
    import torch
    kohya_ss_dict = {}
    
    # Pre-process state_dict to handle PEFT suffixes and basic mapping
    processed_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("base_model.model.", "")
        
        # Handle PEFT suffixes for different network types
        # LoRA: .lora_A.weight, .lora_B.weight
        # LoHa: .hada_w1_a.default, .hada_w1_b.default, .hada_w2_a.default, .hada_w2_b.default
        # LoKr: .lokr_w1.default, .lokr_w2.default
        
        new_key = new_key.replace(".lora_A.weight", ".lora_down.weight")
        new_key = new_key.replace(".lora_B.weight", ".lora_up.weight")
        
        # LoHa mapping
        new_key = new_key.replace(".hada_w1_a.default", ".hada_w1_a.weight")
        new_key = new_key.replace(".hada_w1_b.default", ".hada_w1_b.weight")
        new_key = new_key.replace(".hada_w2_a.default", ".hada_w2_a.weight")
        new_key = new_key.replace(".hada_w2_b.default", ".hada_w2_b.weight")
        
        # LoKr mapping
        new_key = new_key.replace(".lokr_w1.default", ".lokr_w1.weight")
        new_key = new_key.replace(".lokr_w2.default", ".lokr_w2.weight")
        
        # General .default to .weight for any other PEFT types
        if new_key.endswith(".default"):
            new_key = new_key.replace(".default", ".weight")
            
        processed_dict[new_key] = value

    # For Flux, we handle concatenation of Q, K, V to match native checkpoint structure
    if model_type in ["flux1", "flux2"]:
        temp_dict = {}
        for key, value in processed_dict.items():
            new_key = key
            
            # Map text encoders
            if new_key.startswith("text_encoder."):
                new_key = new_key.replace("text_encoder.", "lora_te1_")
            elif new_key.startswith("text_encoder_2."):
                new_key = new_key.replace("text_encoder_2.", "lora_t5_")
            else:
                # Transformer keys
                if new_key.startswith("transformer."):
                    new_key = new_key.replace("transformer.", "lora_unet_")
                elif not new_key.startswith("lora_unet_"):
                    new_key = "lora_unet_" + new_key
                
                # Map blocks
                new_key = new_key.replace("lora_unet_transformer_blocks.", "lora_unet_double_blocks_")
                new_key = new_key.replace("lora_unet_single_transformer_blocks.", "lora_unet_single_blocks_")
                
                # Sub-layer mapping for Flux (Diffusers to Kohya)
                # Double Blocks
                new_key = new_key.replace("img_attn.to_q", "img_attn_q")
                new_key = new_key.replace("img_attn.to_k", "img_attn_k")
                new_key = new_key.replace("img_attn.to_v", "img_attn_v")
                new_key = new_key.replace("txt_attn.to_q", "txt_attn_q")
                new_key = new_key.replace("txt_attn.to_k", "txt_attn_k")
                new_key = new_key.replace("txt_attn.to_v", "txt_attn_v")
                
                # Single Blocks
                new_key = new_key.replace("attn.to_q", "linear1_q")
                new_key = new_key.replace("attn.to_k", "linear1_k")
                new_key = new_key.replace("attn.to_v", "linear1_v")
                
                new_key = new_key.replace("attn.to_out.0", "img_attn_proj")
                new_key = new_key.replace("attn.to_add_out", "txt_attn_proj")
                new_key = new_key.replace("ff.net.0.proj", "linear1_mlp") # For single blocks
                new_key = new_key.replace("ff.net.2", "linear2") # For single blocks
                
                new_key = new_key.replace("ff_context.net.0.proj", "txt_mlp_0")
                new_key = new_key.replace("ff_context.net.2", "txt_mlp_2")
                new_key = new_key.replace("norm1.linear", "img_mod_lin")
                new_key = new_key.replace("norm1_context.linear", "txt_mod_lin")
                new_key = new_key.replace("proj_mlp", "linear1_mlp")
                new_key = new_key.replace("proj_out", "linear2")
                new_key = new_key.replace("norm.linear", "modulation_lin")
            
            # Final dot to underscore for prefix (before the weight suffix)
            # We need to be careful not to replace dots in the weight suffix itself
            suffixes = [".lora_down.weight", ".lora_up.weight", ".hada_w1_a.weight", ".hada_w1_b.weight", 
                        ".hada_w2_a.weight", ".hada_w2_b.weight", ".lokr_w1.weight", ".lokr_w2.weight", ".weight"]
            
            found_suffix = None
            for s in suffixes:
                if new_key.endswith(s):
                    found_suffix = s
                    break
            
            if found_suffix:
                prefix = new_key[:-len(found_suffix)]
                new_key = prefix.replace(".", "_") + found_suffix
            
            temp_dict[new_key] = value

        merged_keys = set()
        # Double blocks img_attn_qkv
        # Note: Concatenation only makes sense for standard LoRA. 
        # For LoHa/LoKr, we keep them separate as they are already complex.
        for i in range(19):
            for suffix_pair in [("lora_down.weight", "lora_up.weight")]:
                down_suffix, up_suffix = suffix_pair
                q_down, k_down, v_down = [f"lora_unet_double_blocks_{i}_img_attn_{x}.{down_suffix}" for x in ["q", "k", "v"]]
                q_up, k_up, v_up = [f"lora_unet_double_blocks_{i}_img_attn_{x}.{up_suffix}" for x in ["q", "k", "v"]]
                
                if all(k in temp_dict for k in [q_down, k_down, v_down, q_up, k_up, v_up]):
                    # Concatenate
                    new_down = f"lora_unet_double_blocks_{i}_img_attn_qkv.{down_suffix}"
                    new_up = f"lora_unet_double_blocks_{i}_img_attn_qkv.{up_suffix}"
                    
                    kohya_ss_dict[new_down] = torch.cat([temp_dict[q_down], temp_dict[k_down], temp_dict[v_down]], dim=0)
                    # For LoRA B (up), we need to block diagonalize to maintain independent updates for Q, K, V
                    # but Kohya's loader expects a single matrix if it's merged.
                    # Actually, Kohya's Flux LoRA often just concatenates them if they were trained as one.
                    # If they were trained separately, block_diag is the mathematically correct way to represent it as one.
                    kohya_ss_dict[new_up] = torch.block_diag(temp_dict[q_up], temp_dict[k_up], temp_dict[v_up])
                    
                    merged_keys.update([q_down, k_down, v_down, q_up, k_up, v_up])
                
                # txt_attn_qkv
                q_down, k_down, v_down = [f"lora_unet_double_blocks_{i}_txt_attn_{x}.{down_suffix}" for x in ["q", "k", "v"]]
                q_up, k_up, v_up = [f"lora_unet_double_blocks_{i}_txt_attn_{x}.{up_suffix}" for x in ["q", "k", "v"]]
                
                if all(k in temp_dict for k in [q_down, k_down, v_down, q_up, k_up, v_up]):
                    new_down = f"lora_unet_double_blocks_{i}_txt_attn_qkv.{down_suffix}"
                    new_up = f"lora_unet_double_blocks_{i}_txt_attn_qkv.{up_suffix}"
                    
                    kohya_ss_dict[new_down] = torch.cat([temp_dict[q_down], temp_dict[k_down], temp_dict[v_down]], dim=0)
                    kohya_ss_dict[new_up] = torch.block_diag(temp_dict[q_up], temp_dict[k_up], temp_dict[v_up])
                    
                    merged_keys.update([q_down, k_down, v_down, q_up, k_up, v_up])

        # Single blocks linear1
        for i in range(38):
            for suffix_pair in [("lora_down.weight", "lora_up.weight")]:
                down_suffix, up_suffix = suffix_pair
                q_down, k_down, v_down, m_down = [f"lora_unet_single_blocks_{i}_linear1_{x}.{down_suffix}" for x in ["q", "k", "v", "mlp"]]
                q_up, k_up, v_up, m_up = [f"lora_unet_single_blocks_{i}_linear1_{x}.{up_suffix}" for x in ["q", "k", "v", "mlp"]]
                
                if all(k in temp_dict for k in [q_down, k_down, v_down, m_down, q_up, k_up, v_up, m_up]):
                    new_down = f"lora_unet_single_blocks_{i}_linear1.{down_suffix}"
                    new_up = f"lora_unet_single_blocks_{i}_linear1.{up_suffix}"
                    
                    kohya_ss_dict[new_down] = torch.cat([temp_dict[q_down], temp_dict[k_down], temp_dict[v_down], temp_dict[m_down]], dim=0)
                    kohya_ss_dict[new_up] = torch.block_diag(temp_dict[q_up], temp_dict[k_up], temp_dict[v_up], temp_dict[m_up])
                    
                    merged_keys.update([q_down, k_down, v_down, m_down, q_up, k_up, v_up, m_up])

        for key, value in temp_dict.items():
            if key not in merged_keys:
                kohya_ss_dict[key] = value

    else:
        # Standard mapping for SDXL, SD1.5, SD3, SD3.5
        for key, value in processed_dict.items():
            new_key = key
            if new_key.startswith("text_encoder."):
                new_key = new_key.replace("text_encoder.", "lora_te1_" if model_type in ["sdxl", "sd3", "sd3.5"] else "lora_te_")
            elif new_key.startswith("text_encoder_2."):
                new_key = new_key.replace("text_encoder_2.", "lora_te2_")
            elif new_key.startswith("text_encoder_3."):
                new_key = new_key.replace("text_encoder_3.", "lora_te3_")
            else:
                # UNet/Transformer keys
                if new_key.startswith("unet."):
                    new_key = new_key.replace("unet.", "lora_unet_")
                elif new_key.startswith("transformer."):
                    new_key = new_key.replace("transformer.", "lora_unet_")
                elif not new_key.startswith("lora_unet_"):
                    new_key = "lora_unet_" + new_key
            
            # Final dot to underscore for prefix
            suffixes = [".lora_down.weight", ".lora_up.weight", ".hada_w1_a.weight", ".hada_w1_b.weight", 
                        ".hada_w2_a.weight", ".hada_w2_b.weight", ".lokr_w1.weight", ".lokr_w2.weight", ".weight"]
            
            found_suffix = None
            for s in suffixes:
                if new_key.endswith(s):
                    found_suffix = s
                    break
            
            if found_suffix:
                prefix = new_key[:-len(found_suffix)]
                new_key = prefix.replace(".", "_") + found_suffix
            
            kohya_ss_dict[new_key] = value

    # Add alpha keys for all layers that need them
    final_dict = {}
    for key, value in kohya_ss_dict.items():
        final_dict[key] = value
        
        # Determine if this is a "down" or "w1" or "w1_a" layer to add alpha
        is_alpha_target = False
        alpha_suffix = None
        
        if ".lora_down.weight" in key:
            is_alpha_target = True
            alpha_suffix = ".lora_down.weight"
        elif ".hada_w1_a.weight" in key:
            is_alpha_target = True
            alpha_suffix = ".hada_w1_a.weight"
        elif ".lokr_w1.weight" in key:
            is_alpha_target = True
            alpha_suffix = ".lokr_w1.weight"
            
        if is_alpha_target:
            alpha_key = key.replace(alpha_suffix, ".alpha")
            # For concatenated layers, alpha should match the concatenated rank
            # But usually network_alpha is what users want
            rank = value.shape[0]
            final_dict[alpha_key] = torch.tensor(float(network_alpha or rank))
            
    return final_dict

def generate_class_images(config: Any, accelerator: Accelerator, status_callback: Optional[Callable]):
    """
    Generates regularization images for DreamBooth if they don't exist.
    """
    if not config.use_prior_preservation or not config.auto_generate_reg_images:
        return

    reg_data_dir = Path(config.reg_data_dir or "project/reg")
    reg_data_dir.mkdir(parents=True, exist_ok=True)
    
    cur_class_images = len(list(reg_data_dir.glob("*.jpg"))) + len(list(reg_data_dir.glob("*.png")))
    if cur_class_images >= config.num_class_images:
        return

    num_new_images = config.num_class_images - cur_class_images
    if status_callback:
        status_callback(0, 0, 0, 0, f"Generating {num_new_images} regularization images...")

    # Load pipeline for generation
    from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline # type: ignore
    import torch
    import time
    
    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    
    # Determine the correct pipeline class based on the model type
    # We default to DiffusionPipeline but try to be specific for SDXL/SD1.5
    pipeline_class = DiffusionPipeline
    model_type = getattr(config, "model_type", "sdxl").lower()
    
    if "sdxl" in model_type:
        pipeline_class = StableDiffusionXLPipeline
    elif "sd15" in model_type or "sd1.5" in model_type:
        pipeline_class = StableDiffusionPipeline

    try:
        info(f"Loading generation pipeline using {pipeline_class.__name__}...")
        pipeline: Any = pipeline_class.from_single_file( # type: ignore
            config.base_model_name,
            torch_dtype=torch_dtype,
            safety_checker=None,
            use_safetensors=True
        )
    except AttributeError as e:
        error("Failed to load pipeline: 'from_single_file' is missing. This usually means 'omegaconf' is not installed.")
        raise RuntimeError("Missing dependency: omegaconf. Please restart the application to install it.") from e
    except Exception as e:
        error(f"Failed to load generation pipeline: {e}")
        raise e

    pipeline.set_progress_bar_config(disable=True)
    
    # Initialize Memory Manager and optimize
    from system.hardware import MemoryManager, apply_unified_memory_optimizations
    memory_manager = MemoryManager()
    
    # Apply unified optimizations (slicing, tiling, etc.)
    apply_unified_memory_optimizations(pipeline, accelerator.device, torch_dtype)
    
    # Prepare UNet with adaptive memory management (CPU offload/swap)
    # This handles device placement automatically
    pipeline = memory_manager.prepare_pipeline_memory(
        pipeline,
        width=config.resolution,
        height=config.resolution,
        num_inference_steps=config.reg_infer_steps or 20
    )
    
    # Optimize pipeline if possible
    try:
        import xformers
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass

    # Set scheduler if requested
    if config.reg_scheduler:
        from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
        from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
        
        sched_name = config.reg_scheduler.lower()
        if sched_name == "euler_a":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        elif sched_name == "dpm++_2m_karras":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
        elif sched_name == "euler":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        # Default fallback is whatever the model has

    if config.reg_seed is not None and config.reg_seed != -1:
        generator = torch.Generator(device=accelerator.device).manual_seed(config.reg_seed)
    else:
        generator = None
        
    # Warn if class prompt looks suspicious (same as instance prompt)
    if config.instance_prompt and config.class_prompt and config.instance_prompt.strip() == config.class_prompt.strip():
        from system.log import warning
        warning(f"Class Prompt is identical to Instance Prompt ('{config.class_prompt}'). This defeats the purpose of Prior Preservation! You should use a generic class name (e.g. 'girl') for Class Prompt.")

    # Generate
    info(f"Generating {num_new_images} regularization images...")
    
    for i in tqdm(range(num_new_images), desc="Generating Reg Images"):
        with torch.no_grad():
            image = pipeline(
                prompt=config.class_prompt,
                negative_prompt=config.reg_negative_prompt,
                num_inference_steps=config.reg_infer_steps,
                guidance_scale=config.reg_guidance_scale,
                generator=generator,
                output_type="pil"
            ).images[0]
            
        save_path = reg_dir / f"reg_{i:05d}.png"
        image.save(save_path)
        
    del pipeline
    flush()
    info("Regularization image generation complete.")

def cache_latents_to_disk(vae, dataset, config, accelerator, status_callback: Optional[Callable] = None):
    """
    Pre-computes latents for all instance images and saves them to disk.
    """
    cache_dir = get_latents_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # New layout uses subfolders for readability / collision avoidance.
    (cache_dir / "instance").mkdir(parents=True, exist_ok=True)
    (cache_dir / "class").mkdir(parents=True, exist_ok=True)

    index_path = cache_dir / "index.jsonl"
    
    info(f"Caching latents to {cache_dir}...")

    if status_callback:
        try:
            status_callback(0, 1, 0.0, 0, f"Caching latents to disk ({cache_dir})...", phase="caching_latents")
        except TypeError:
            # Backward-compat if callback doesn't accept phase yet
            status_callback(0, 1, 0.0, 0, f"Caching latents to disk ({cache_dir})...")
    
    vae.to(accelerator.device, dtype=torch.float32)
    vae.eval()

    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    paths = list(dataset.instance_images_paths)
    batch_size = config.vae_batch_size or config.batch_size

    class_paths: List[Path] = []
    if getattr(dataset, "class_images_paths", None):
        class_paths = list(dataset.class_images_paths)  # type: ignore[arg-type]
    
    # Process in chunks
    def _gather_missing(to_check: List[Path], kind: str, root: Path) -> List[Path]:
        missing: List[Path] = []
        for p in to_check:
            candidates = _latent_cache_paths(cache_dir, kind, root, p)
            if not any(c.exists() for c in candidates):
                missing.append(p)
        return missing

    instance_missing = _gather_missing(paths, "instance", dataset.instance_data_root)
    class_missing = _gather_missing(class_paths, "class", dataset.class_data_root) if class_paths and dataset.class_data_root else []

    work: List[tuple[str, Path, Path]] = []
    work.extend([("instance", p, dataset.instance_data_root) for p in instance_missing])
    if class_missing and dataset.class_data_root:
        work.extend([("class", p, dataset.class_data_root) for p in class_missing])

    if not work:
        info("Latents cache already present. Skipping VAE caching.")
        if status_callback:
            try:
                status_callback(1, 1, 0.0, 0, "Latents cache already present.", phase="caching_latents")
            except TypeError:
                status_callback(1, 1, 0.0, 0, "Latents cache already present.")
        accelerator.wait_for_everyone()
        return

    total_items = len(work)
    processed_items = 0

    if status_callback:
        try:
            status_callback(0, total_items, 0.0, 0, None, phase="caching_latents")
        except TypeError:
            status_callback(0, total_items, 0.0, 0, None)

    for i in tqdm(range(0, len(work), batch_size), desc="Caching Latents"):
        batch_items = work[i : i + batch_size]
        batch_images = []
        batch_metadata = []
        
        for kind, path, root in batch_items:
            try:
                img = Image.open(path)
                img = exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                original_size = img.size
                cropped_img, crop_box, _ = _center_crop_to_square_div8(img, dataset.size)
                
                if cropped_img.size[0] != dataset.size:
                    cropped_img = cropped_img.resize((dataset.size, dataset.size), Image.Resampling.LANCZOS)
                
                # No augmentation for caching
                tensor = dataset.final_transforms(cropped_img)
                batch_images.append(tensor)
                batch_metadata.append({
                    "kind": kind,
                    "root": root,
                    "original_size": (original_size[1], original_size[0]),
                    "crop_top_left": (crop_box[1], crop_box[0]),
                    "path": path
                })
            except Exception as e:
                print(f"Error processing {path} for cache: {e}")
                continue
            
        if not batch_images:
            continue

        # Stack
        pixel_values = torch.stack(batch_images).to(accelerator.device, dtype=torch.float32)
        
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            shift_factor = getattr(vae.config, "shift_factor", None)
            if shift_factor is not None:
                latents = (latents - shift_factor) * vae.config.scaling_factor
            else:
                latents = latents * vae.config.scaling_factor
            
        # Save
        for j, latent in enumerate(latents):
            meta = batch_metadata[j]
            save_path = _latent_cache_paths(cache_dir, meta["kind"], meta["root"], meta["path"])[0]
            data = {
                "latents": latent.cpu(),
                "original_size": meta["original_size"],
                "crop_top_left": meta["crop_top_left"]
            }
            torch.save(data, save_path)

            # Write reverse-lookup info for humans/tools.
            try:
                entry = {
                    "kind": meta["kind"],
                    "key": _latent_cache_key(meta["root"], meta["path"]),
                    "rel_path": _latent_cache_relpath(meta["root"], meta["path"]),
                    "cache_path": str(save_path.relative_to(cache_dir)).replace("\\", "/"),
                    "original_size": list(meta["original_size"]),
                    "crop_top_left": list(meta["crop_top_left"]),
                }
                with open(index_path, "a", encoding="utf-8") as f:
                    import json

                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                # Index is best-effort; cache files remain authoritative.
                pass

        processed_items += len(latents)
        if status_callback:
            try:
                status_callback(processed_items, total_items, 0.0, 0, None, phase="caching_latents")
            except TypeError:
                status_callback(processed_items, total_items, 0.0, 0, None)
            
    info(f"Cached latents: instance={len(instance_missing)} class={len(class_missing)}")
    flush()
    if status_callback:
        try:
            status_callback(total_items, total_items, 0.0, 0, "Latent caching complete.", phase="caching_latents")
        except TypeError:
            status_callback(total_items, total_items, 0.0, 0, "Latent caching complete.")
    accelerator.wait_for_everyone()
