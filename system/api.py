from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json

from system import coordinator_settings as cs
from system.context import MODEL_MANAGER, DATASET_MANAGER, TRAINING_MANAGER
from system.training.schema import TrainingConfig
from system.hardware import get_system_info
from system import updater

# Request Models
class CaptionUpdate(BaseModel):
    image_name: str
    caption: str

class MetadataUpdate(BaseModel):
    metadata: Dict[str, str]

class ConversionRequest(BaseModel):
    file_path: str
    model_type: str
    target_format: str
    output_name: Optional[str] = None

class StartTrainingRequest(BaseModel):
    config: TrainingConfig


class AutoAdjustRequest(BaseModel):
    config: TrainingConfig

class UpdateApplyRequest(BaseModel):
    selected_paths: Optional[List[str]] = None

class LocalizationSwitchRequest(BaseModel):
    lang_code: str

class AugmentationPreviewRequest(BaseModel):
    resolution: int = 1024
    dataset_path: str = "project/dataset"
    count: int = 100
    # New individual augmentation values
    crop_jitter: float = 0.0
    random_flip: float = 0.0
    random_brightness: float = 0.0
    random_contrast: float = 0.0
    random_saturation: float = 0.0
    random_hue: float = 0.0

def init_app(root_path: Path):
    app = FastAPI(title="Onika Trainer")
    
    # API Endpoints
    
    @app.get("/api/status")
    async def get_status():
        return {
            "status": "running",
            "training": TRAINING_MANAGER.get_status(),
            "system": get_system_info()
        }

    @app.get("/api/system/suggestions")
    async def get_suggestions():
        from system.hardware import suggest_optimizations
        info = get_system_info()
        if info["gpu"]["has_cuda"]:
            vram = info["gpu"]["gpus"][0]["total_memory"]
            return suggest_optimizations(vram)
        return suggest_optimizations(0)

    @app.get("/api/localization")
    async def get_localization():
        # Reload settings to get current language
        settings = cs.reload_settings()
        current_lang = settings.get("language", "en")
        
        # Get available languages
        loc_dir = root_path / "config" / "localizations"
        languages = []
        if loc_dir.exists():
            for f in loc_dir.glob("*.json"):
                lang_name = f.stem.upper()
                try:
                    with open(f, "r", encoding="utf-8") as lf:
                        ldata = json.load(lf)
                        if "language_name" in ldata:
                            lang_name = ldata["language_name"]
                except:
                    pass
                languages.append({"code": f.stem, "name": lang_name})
        
        # Load current language file
        lang_file = loc_dir / f"{current_lang}.json"
        localization_data = {}
        if lang_file.exists():
            try:
                with open(lang_file, "r", encoding="utf-8") as f:
                    localization_data = json.load(f)
            except Exception as e:
                print(f"Error loading localization file: {e}")
        
        return {
            "localization": localization_data,
            "active": current_lang,
            "languages": languages
        }

    @app.post("/api/localization/switch")
    async def switch_language(data: LocalizationSwitchRequest):
        # Update config file
        config_path = root_path / "config" / "configs.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            config["language"] = data.lang_code
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
                
            # Reload settings in memory
            cs.reload_settings()
            
            # Return new localization data
            loc_dir = root_path / "config" / "localizations"
            lang_file = loc_dir / f"{data.lang_code}.json"
            localization_data = {}
            if lang_file.exists():
                with open(lang_file, "r", encoding="utf-8") as f:
                    localization_data = json.load(f)
            
            return {
                "localization": localization_data,
                "active": data.lang_code
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/models")
    async def get_models():
        return MODEL_MANAGER.get_models()

    @app.get("/api/dataset/images")
    async def get_dataset_images():
        return DATASET_MANAGER.get_images()

    @app.get("/api/dataset/scan")
    async def scan_dataset():
        return DATASET_MANAGER.scan_dataset()

    @app.post("/api/dataset/caption")
    async def update_caption(data: CaptionUpdate):
        try:
            DATASET_MANAGER.update_caption(data.image_name, data.caption)
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/training/status")
    async def get_training_status():
        return TRAINING_MANAGER.get_status()

    @app.get("/api/training/config")
    async def get_training_config():
        # Try current project config first, then fallback to default
        config_path = root_path / "project" / "current_training_config.json"
        if not config_path.exists():
            config_path = root_path / "config" / "training.json"
            
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading training config: {e}")
        
        # Default config
        return TrainingConfig(
            base_model_name="", # User must select
            dataset_path=str(root_path / "project" / "dataset"),
            output_dir=str(root_path / "project" / "outputs")
        ).model_dump()

    @app.post("/api/training/config")
    async def save_training_config(config: TrainingConfig):
        config_path = root_path / "project" / "current_training_config.json"
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config.model_dump(), f, indent=4)
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/training/start")
    async def start_training(config: TrainingConfig):
        # Save config before starting
        try:
            config_path = root_path / "project" / "current_training_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config.model_dump(), f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save current training config: {e}")

        try:
            TRAINING_MANAGER.start_training(config)
            return {"status": "started"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/training/auto_adjust")
    async def auto_adjust(req: AutoAdjustRequest):
        """Return a preset-like patch tuned for the current dataset + chosen architecture.

        IMPORTANT: This does not select the model or architecture. It only recommends config values.
        """
        from system.training.adjust import recommend_training_patch

        try:
            return recommend_training_patch(req.config, root_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/training/stop")
    async def stop_training():
        TRAINING_MANAGER.stop_training()
        return {"status": "stopping"}

    @app.get("/api/outputs")
    async def get_outputs(dir: Optional[str] = None):
        target_dir = Path(dir) if dir else root_path / "project" / "outputs"
        if not target_dir.exists():
            return []
        
        files = []
        for f in target_dir.glob("*.safetensors"):
            files.append({
                "name": f.name,
                "path": str(f.absolute()),
                "size": f.stat().st_size
            })
        return files

    @app.get("/api/presets")
    async def get_presets():
        presets_dir = root_path / "presets"
        if not presets_dir.exists():
            return []
        return [f.stem for f in presets_dir.glob("*.json")]

    @app.get("/api/presets/{name}")
    async def get_preset(name: str):
        preset_path = root_path / "presets" / f"{name}.json"
        if not preset_path.exists():
            raise HTTPException(status_code=404, detail="Preset not found")
        with open(preset_path, "r") as f:
            return json.load(f)

    @app.post("/api/presets/{name}")
    async def save_preset(name: str, config: Dict[str, Any]):
        presets_dir = root_path / "presets"
        presets_dir.mkdir(exist_ok=True)
        preset_path = presets_dir / f"{name}.json"
        with open(preset_path, "w") as f:
            json.dump(config, f, indent=4)
        return {"status": "success"}

    @app.delete("/api/presets/{name}")
    async def delete_preset(name: str):
        preset_path = root_path / "presets" / f"{name}.json"
        if preset_path.exists():
            preset_path.unlink()
        return {"status": "success"}

    @app.get("/api/metadata/{filename}")
    async def get_metadata(filename: str):
        return {"name": filename, "type": "lora", "base_model": "sdxl"}

    @app.post("/api/metadata/{filename}")
    async def save_metadata(filename: str, data: MetadataUpdate):
        return {"status": "success"}

    @app.post("/api/tools/convert")
    async def convert_model(req: ConversionRequest):
        return {"status": "success", "output_path": req.output_name or "converted.safetensors"}

    # Update Endpoints
    @app.get("/api/updates/check")
    async def check_updates():
        return updater.check_for_updates()

    @app.post("/api/updates/apply")
    async def apply_updates(data: UpdateApplyRequest):
        return updater.apply_updates(data.selected_paths)

    # Augmentation Preview - Session Storage
    _aug_preview_cache = {}  # session_id -> {images: [...], expires: timestamp}
    
    @app.post("/api/augmentation/preview")
    async def augmentation_preview(req: AugmentationPreviewRequest):
        """Generate augmentation preview metadata (no image data, just metadata + session ID)."""
        import random
        import io
        import uuid
        import time
        from PIL import Image, ImageDraw
        from PIL.ImageOps import exif_transpose
        
        dataset_dir = Path(req.dataset_path)
        if not dataset_dir.is_absolute():
            dataset_dir = root_path / dataset_dir
        
        if not dataset_dir.exists():
            raise HTTPException(status_code=400, detail=f"Dataset path not found: {req.dataset_path}")
        
        # Get all valid images
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        all_images = [p for p in dataset_dir.iterdir() if p.suffix.lower() in valid_exts]
        
        if not all_images:
            raise HTTPException(status_code=400, detail="No images found in dataset")
        
        # Sample random images (up to count)
        sample_count = min(req.count, len(all_images))
        sampled_images = random.sample(all_images, sample_count)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        results = []
        cached_images = []
        target_size = req.resolution
        preview_size = 384  # Thumbnail size
        
        # Make resolution divisible by 8
        target_size = (target_size // 8) * 8
        if target_size == 0:
            target_size = 8
        
        for idx, img_path in enumerate(sampled_images):
            try:
                # Load image
                img = Image.open(img_path)
                img = exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                original_size = img.size  # (w, h)
                augmentations_applied = []
                
                w, h = original_size
                
                # Crop Jitter - value is the jitter amount (0 = no jitter, 1 = full jitter)
                if req.crop_jitter > 0:
                    # crop_jitter of 0.2 means crop scale between 0.8 and 1.0
                    min_scale = 1.0 - req.crop_jitter
                    actual_scale = random.uniform(min_scale, 1.0)
                    scaled_size = int(min(w, h) * actual_scale)
                    augmentations_applied.append(f"Crop {actual_scale:.0%}")
                else:
                    scaled_size = min(w, h)
                    actual_scale = 1.0
                
                # Make divisible by 8
                final_crop_size = (scaled_size // 8) * 8
                if final_crop_size == 0:
                    final_crop_size = 8
                
                # Center crop coordinates
                crop_left = (w - final_crop_size) // 2
                crop_top = (h - final_crop_size) // 2
                crop_right = crop_left + final_crop_size
                crop_bottom = crop_top + final_crop_size
                crop_box = (crop_left, crop_top, crop_right, crop_bottom)
                
                # Apply crop
                cropped_img = img.crop(crop_box)
                
                # Resize to target
                if cropped_img.size[0] != target_size:
                    resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                    cropped_img = cropped_img.resize((target_size, target_size), resample_method)
                
                # Random Flip
                flipped = False
                if req.random_flip > 0 and random.random() < req.random_flip:
                    transpose_method = Image.Transpose.FLIP_LEFT_RIGHT if hasattr(Image, 'Transpose') else Image.FLIP_LEFT_RIGHT
                    cropped_img = cropped_img.transpose(transpose_method)
                    flipped = True
                    augmentations_applied.append("Flip")
                
                # Individual color augmentations
                from PIL import ImageEnhance
                
                # Random Brightness
                if req.random_brightness > 0:
                    factor = 1.0 + random.uniform(-req.random_brightness, req.random_brightness)
                    enhancer = ImageEnhance.Brightness(cropped_img)
                    cropped_img = enhancer.enhance(factor)
                    augmentations_applied.append("Brightness")
                
                # Random Contrast
                if req.random_contrast > 0:
                    factor = 1.0 + random.uniform(-req.random_contrast, req.random_contrast)
                    enhancer = ImageEnhance.Contrast(cropped_img)
                    cropped_img = enhancer.enhance(factor)
                    augmentations_applied.append("Contrast")
                
                # Random Saturation
                if req.random_saturation > 0:
                    factor = 1.0 + random.uniform(-req.random_saturation, req.random_saturation)
                    enhancer = ImageEnhance.Color(cropped_img)
                    cropped_img = enhancer.enhance(factor)
                    augmentations_applied.append("Saturation")
                
                # Random Hue
                if req.random_hue > 0:
                    import colorsys
                    import numpy as np
                    img_array = np.array(cropped_img).astype(np.float32) / 255.0
                    hue_shift = random.uniform(-req.random_hue, req.random_hue)
                    result = np.zeros_like(img_array)
                    for i in range(img_array.shape[0]):
                        for j in range(img_array.shape[1]):
                            r, g, b = img_array[i, j]
                            h, s, v = colorsys.rgb_to_hsv(r, g, b)
                            h = (h + hue_shift) % 1.0
                            r, g, b = colorsys.hsv_to_rgb(h, s, v)
                            result[i, j] = [r, g, b]
                    cropped_img = Image.fromarray((result * 255).astype(np.uint8))
                    augmentations_applied.append("Hue")
                
                # Create preview image with crop overlay on original
                scale_factor = preview_size / max(original_size)
                display_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
                resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                display_img = img.resize(display_size, resample_method)
                
                scaled_crop = (
                    int(crop_left * scale_factor),
                    int(crop_top * scale_factor),
                    int(crop_right * scale_factor),
                    int(crop_bottom * scale_factor)
                )
                
                # Semi-transparent overlay outside crop area
                overlay = Image.new('RGBA', display_img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                
                overlay_draw.rectangle([0, 0, display_img.size[0], scaled_crop[1]], fill=(0, 0, 0, 120))
                overlay_draw.rectangle([0, scaled_crop[3], display_img.size[0], display_img.size[1]], fill=(0, 0, 0, 120))
                overlay_draw.rectangle([0, scaled_crop[1], scaled_crop[0], scaled_crop[3]], fill=(0, 0, 0, 120))
                overlay_draw.rectangle([scaled_crop[2], scaled_crop[1], display_img.size[0], scaled_crop[3]], fill=(0, 0, 0, 120))
                
                overlay_draw.rectangle(scaled_crop, outline=(205, 255, 0, 255), width=2)
                
                display_img = display_img.convert('RGBA')
                display_img = Image.alpha_composite(display_img, overlay)
                display_img = display_img.convert('RGB')
                
                # Store JPEG bytes in cache (not base64)
                aug_buffer = io.BytesIO()
                resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                aug_thumb = cropped_img.resize((preview_size, preview_size), resample_method)
                aug_thumb.save(aug_buffer, format='JPEG', quality=85)
                aug_bytes = aug_buffer.getvalue()
                
                orig_buffer = io.BytesIO()
                display_img.save(orig_buffer, format='JPEG', quality=85)
                orig_bytes = orig_buffer.getvalue()
                
                # Cache the raw bytes
                cached_images.append({
                    "original_bytes": orig_bytes,
                    "augmented_bytes": aug_bytes
                })
                
                # Return only metadata (no image data)
                results.append({
                    "id": idx,
                    "filename": img_path.name,
                    "original_size": list(original_size),
                    "crop_box": list(crop_box),
                    "augmentations": augmentations_applied,
                    "flipped": flipped,
                    "crop_scale": actual_scale if actual_scale < 1.0 else 1.0
                })
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clean up old sessions (older than 10 minutes)
        current_time = time.time()
        expired = [sid for sid, data in _aug_preview_cache.items() if data["expires"] < current_time]
        for sid in expired:
            del _aug_preview_cache[sid]
        
        # Store new session (expires in 10 minutes)
        _aug_preview_cache[session_id] = {
            "images": cached_images,
            "expires": current_time + 600
        }
        
        return {
            "status": "success",
            "session_id": session_id,
            "count": len(results),
            "total_dataset": len(all_images),
            "images": results
        }
    
    @app.get("/api/augmentation/preview/{session_id}/{index}/{img_type}")
    async def get_augmentation_image(session_id: str, index: int, img_type: str):
        """Return raw JPEG bytes for a specific preview image."""
        from fastapi.responses import Response
        
        if session_id not in _aug_preview_cache:
            raise HTTPException(status_code=404, detail="Session expired or not found")
        
        session = _aug_preview_cache[session_id]
        
        if index < 0 or index >= len(session["images"]):
            raise HTTPException(status_code=404, detail="Image index out of range")
        
        img_data = session["images"][index]
        
        if img_type == "original":
            return Response(content=img_data["original_bytes"], media_type="image/jpeg")
        elif img_type == "augmented":
            return Response(content=img_data["augmented_bytes"], media_type="image/jpeg")
        else:
            raise HTTPException(status_code=400, detail="Invalid image type. Use 'original' or 'augmented'")

    # Serve UI
    ui_path = root_path / "ui"
    
    @app.get("/")
    async def read_index():
        return FileResponse(ui_path / "index.html")

    app.mount("/ui", StaticFiles(directory=str(ui_path)), name="ui")
    
    return app
