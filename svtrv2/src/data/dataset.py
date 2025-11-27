"""Simple dataset for SVTRv2 training from JSONL format"""
import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F


class SVTRv2Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        jsonl_path=None,
        jsonl_paths=None,
        transforms=None,
        base_h=32,
        base_shape=None,
        padding=True,
        max_text_length=25,
    ):
        """
        Args:
            data_dir: Root directory containing images
            jsonl_path: Path to single JSONL file (for backward compatibility)
            jsonl_paths: List of paths to JSONL files (takes precedence over jsonl_path)
            transforms: List of transform functions
            base_h: Base height for resizing
            base_shape: List of [width, height] shapes for multi-scale
            padding: Whether to pad images
            max_text_length: Maximum text length
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        
        # Determine which JSONL files to load
        if jsonl_paths is not None:
            # Multiple JSONL files provided
            if isinstance(jsonl_paths, (str, Path)):
                jsonl_paths = [jsonl_paths]
            jsonl_paths = [Path(p) if not isinstance(p, Path) else p for p in jsonl_paths]
        elif jsonl_path is not None:
            # Single JSONL file provided (backward compatibility)
            jsonl_paths = [Path(jsonl_path)]
        else:
            # Default to dataset.jsonl in data_dir
            jsonl_paths = [self.data_dir / "dataset.jsonl"]
        
        # Load data from all JSONL files
        self.samples = []
        for jsonl_path in jsonl_paths:
            if not jsonl_path.exists():
                continue
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    filename = item["filename"]
                    text = item["text"]
                    
                    # Handle relative paths
                    if filename.startswith("imgs/"):
                        img_path = self.data_dir / filename
                    else:
                        img_path = self.data_dir / "imgs" / filename
                    
                    if img_path.exists():
                        self.samples.append({"image_path": str(img_path), "label": text})
        
        self.transforms = transforms or []
        self.base_h = base_h
        self.base_shape = base_shape or [[64, 32], [96, 32], [128, 32], [160, 32]]
        self.padding = padding
        self.max_text_length = max_text_length
        
        # Image normalization
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.interpolation = T.InterpolationMode.BICUBIC

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = sample["image_path"]
        label = sample["label"]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # If image loading fails, return a random sample
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        
        # Apply transforms
        data = {"image": img, "label": label}
        for transform in self.transforms:
            data = transform(data)
            if data is None:
                return self.__getitem__(random.randint(0, len(self.samples) - 1))
        
        # Resize and normalize
        img = data["image"]
        w, h = img.size
        
        # Simple resize: maintain aspect ratio, pad if needed
        target_h = self.base_h
        target_w = int(w * target_h / h)
        
        # Resize
        img = F.resize(img, (target_h, target_w), interpolation=self.interpolation)
        
        # Pad to fixed width if needed (use max width from base_shape)
        max_w = max([s[0] for s in self.base_shape])
        if target_w < max_w and self.padding:
            pad_w = max_w - target_w
            # Random padding position
            if random.random() < 0.5:
                img = F.pad(img, [0, 0, pad_w, 0], fill=0.0)
            else:
                img = F.pad(img, [pad_w, 0, 0, 0], fill=0.0)
        elif target_w > max_w:
            # Crop if too wide
            img = F.crop(img, 0, 0, target_h, max_w)
        
        # Normalize
        img = self.normalize(img)
        
        return {
            "image": img,
            "label": label,
            "label_length": len(label),
        }

