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

    @app.get("/api/models")
    async def get_models():
        return MODEL_MANAGER.get_models()

    @app.get("/api/dataset/images")
    async def get_dataset_images():
        return DATASET_MANAGER.get_images()

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

    # Serve UI
    ui_path = root_path / "ui"
    
    @app.get("/")
    async def read_index():
        return FileResponse(ui_path / "index.html")

    app.mount("/ui", StaticFiles(directory=str(ui_path)), name="ui")
    
    return app
