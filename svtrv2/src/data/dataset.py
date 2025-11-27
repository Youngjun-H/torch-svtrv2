"""Simple dataset for SVTRv2 training from JSONL format"""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from .label_encode import CTCLabelEncoder


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
        character_dict_path=None,
        use_space_char=False,
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
                    
                    # Handle relative paths (imgs/, train/, val/, etc.)
                    if filename.startswith(("imgs/", "train/", "val/")):
                        img_path = self.data_dir / filename
                    else:
                        # Try imgs/ first, then try as direct path
                        img_path = self.data_dir / "imgs" / filename
                        if not img_path.exists():
                            img_path = self.data_dir / filename
                    
                    if img_path.exists():
                        self.samples.append({"image_path": str(img_path), "label": text})
        
        self.transforms = transforms or []
        self.base_h = base_h
        self.base_shape = base_shape or [[64, 32], [96, 32], [128, 32], [160, 32]]
        self.padding = padding
        self.max_text_length = max_text_length
        
        # Label encoder
        self.label_encoder = CTCLabelEncoder(
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            max_text_length=max_text_length,
        )
        
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
        
        # Resize to target height while maintaining aspect ratio
        target_h = self.base_h
        target_w = int(w * target_h / h)
        
        # Resize
        img = F.resize(img, (target_h, target_w), interpolation=self.interpolation)
        
        # Optional: Limit max width during training (for memory efficiency)
        # Padding will be handled by collate_fn to ensure batch consistency
        max_w = max([s[0] for s in self.base_shape]) if self.base_shape else None
        if max_w and target_w > max_w:
            # Crop if too wide (only during training)
            img = F.crop(img, 0, 0, target_h, max_w)
        
        # Normalize
        img = self.normalize(img)
        
        # Encode label to indices
        label_indices = self.label_encoder.encode(label)
        if label_indices is None:
            # If encoding fails, return a random sample
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        
        # Calculate actual label length (original text length before padding)
        # The encoded label is padded to max_text_length, so we need to find the actual length
        # by counting non-blank tokens before padding
        actual_length = len(label)  # Original text length
        # Ensure at least 1 (CTC requires non-zero length)
        actual_length = max(1, actual_length)
        
        return {
            "image": img,
            "label": label,  # Keep original string for metrics
            "label_indices": torch.tensor(label_indices, dtype=torch.long),
            "label_length": torch.tensor(actual_length, dtype=torch.long),
        }

