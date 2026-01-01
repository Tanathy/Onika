import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
from PIL import Image, ImageStat

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
    
    def scan_dataset(self) -> Dict[str, Any]:
        """Scans the dataset and returns statistics."""
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        total_images = 0
        no_tag_images = 0
        all_tags = set()
        min_dim = [float('inf'), float('inf')]
        max_dim = [0, 0]
        
        # Graph Data
        ar_buckets = [0] * 21  # -1.0 to 1.0 log scale (0.1 steps)
        brightness_buckets = [0] * 21 # -1.0 to 1.0 (0.1 steps)

        if not self.dataset_dir.exists():
            return {
                "total_images": 0,
                "no_tag_images": 0,
                "total_tags": 0,
                "min_dim": [0, 0],
                "max_dim": [0, 0],
                "ar_distribution": ar_buckets,
                "brightness_distribution": brightness_buckets
            }

        files = [f for f in self.dataset_dir.iterdir() if f.suffix.lower() in valid_exts]
        total_images = len(files)
        
        # Sampling for brightness (max 2000 images for speed)
        import random
        import math
        
        sample_indices = set(range(total_images))
        if total_images > 2000:
            sample_indices = set(random.sample(range(total_images), 2000))

        for idx, file_path in enumerate(files):
            # Dimensions & AR (Fast)
            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                    min_dim[0] = min(min_dim[0], w)
                    min_dim[1] = min(min_dim[1], h)
                    max_dim[0] = max(max_dim[0], w)
                    max_dim[1] = max(max_dim[1], h)
                    
                    # Aspect Ratio Log2
                    # log2(1) = 0 (Square)
                    # log2(2) = 1 (2:1 Landscape)
                    # log2(0.5) = -1 (1:2 Portrait)
                    ar_log = math.log2(w / h)
                    # Clamp to -1.0 ... 1.0 for the graph range (covers 1:2 to 2:1)
                    # Map -1.0..1.0 to 0..20
                    ar_idx = int((max(-1.0, min(1.0, ar_log)) + 1.0) * 10)
                    if 0 <= ar_idx < 21:
                        ar_buckets[ar_idx] += 1

                    # Brightness (Slower, only for sampled)
                    if idx in sample_indices:
                        # Resize for speed
                        img_small = img.convert('L').resize((128, 128))
                        # Calculate mean brightness
                        stat = ImageStat.Stat(img_small)
                        mean_brightness = stat.mean[0]
                        # Normalize 0..255 -> -1..1
                        # 0 -> -1, 127.5 -> 0, 255 -> 1
                        balance = (mean_brightness - 127.5) / 127.5
                        # Map -1.0..1.0 to 0..20
                        b_idx = int((max(-1.0, min(1.0, balance)) + 1.0) * 10)
                        if 0 <= b_idx < 21:
                            brightness_buckets[b_idx] += 1

            except Exception:
                pass

            # Tags
            txt_path = file_path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            tags = [t.strip() for t in content.split(",") if t.strip()]
                            for tag in tags:
                                all_tags.add(tag)
                        else:
                            no_tag_images += 1
                except Exception:
                    no_tag_images += 1
            else:
                no_tag_images += 1

        return {
            "total_images": total_images,
            "no_tag_images": no_tag_images,
            "total_tags": len(all_tags),
            "min_dim": [0, 0] if total_images == 0 else min_dim,
            "max_dim": [0, 0] if total_images == 0 else max_dim,
            "ar_distribution": ar_buckets,
            "brightness_distribution": brightness_buckets
        }
