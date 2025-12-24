import os
from pathlib import Path
from typing import List, Dict, Optional
import json
from PIL import Image

class DatasetManager:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.dataset_dir = root_path / "project" / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def get_images(self) -> List[Dict[str, str]]:
        """Returns a list of images and their captions."""
        images = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        
        if not self.dataset_dir.exists():
            return []

        for file_path in self.dataset_dir.iterdir():
            if file_path.suffix.lower() in valid_exts:
                # Look for corresponding text file
                txt_path = file_path.with_suffix(".txt")
                caption = ""
                if txt_path.exists():
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            caption = f.read().strip()
                    except Exception:
                        pass
                
                images.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "caption": caption
                })
        return images

    def update_caption(self, image_name: str, caption: str):
        """Updates the caption for a specific image."""
        image_path = self.dataset_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_name} not found")
        
        txt_path = image_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)

    def get_dataset_path(self) -> str:
        return str(self.dataset_dir.absolute())
