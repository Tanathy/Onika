import os
from pathlib import Path
from typing import List, Dict, Optional

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_model = None

    def get_models(self) -> List[Dict[str, str]]:
        """Scans the models directory for supported model files."""
        if not self.models_dir.exists():
            return []
        
        models = []
        # Supported extensions
        extensions = {".safetensors", ".ckpt", ".pt"}
        
        for file_path in self.models_dir.glob("*"):
            if file_path.suffix.lower() in extensions:
                models.append({
                    "name": file_path.name,
                    "path": str(file_path.absolute()),
                    "size": file_path.stat().st_size
                })
        return models

    def get_info(self):
        """Returns info about the currently loaded model (if any)."""
        # For now, we don't keep the model loaded in memory in this process 
        # if we are just orchestrating training.
        return None
