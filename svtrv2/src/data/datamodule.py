"""Lightning DataModule for SVTRv2"""
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from .dataset import SVTRv2Dataset


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized images in a batch.
    Pads all images to the same width (max width in batch).
    """
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    label_indices = [item["label_indices"] for item in batch]
    label_lengths = [item["label_length"] for item in batch]
    
    # Find max width in batch
    max_width = max(img.shape[2] for img in images)
    max_height = max(img.shape[1] for img in images)
    
    # Pad all images to same size
    padded_images = []
    for img in images:
        _, h, w = img.shape
        pad_w = max_width - w
        pad_h = max_height - h
        
        # Pad: (left, right, top, bottom)
        if pad_w > 0 or pad_h > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        padded_images.append(img)
    
    # Stack tensors
    images_tensor = torch.stack(padded_images, dim=0)
    label_indices_tensor = torch.stack(label_indices, dim=0)
    label_lengths_tensor = torch.stack(label_lengths, dim=0)
    
    return {
        "image": images_tensor,
        "label": labels,
        "label_indices": label_indices_tensor,
        "label_length": label_lengths_tensor,
    }


class SVTRv2DataModule(L.LightningDataModule):
    """Simple DataModule for SVTRv2 training"""

    def __init__(
        self,
        train_data_dir,
        train_jsonl_paths=None,
        val_data_dir=None,
        val_jsonl_paths=None,
        batch_size=32,
        num_workers=4,
        train_transforms=None,
        val_transforms=None,
        base_h=32,
        base_shape=None,
        padding=True,
        max_text_length=25,
        character_dict_path=None,
        use_space_char=False,
    ):
        """
        Args:
            train_data_dir: Directory containing training data
            train_jsonl_paths: List of paths to training JSONL files (optional)
            val_data_dir: Directory containing validation data (optional)
            val_jsonl_paths: List of paths to validation JSONL files (optional)
            batch_size: Batch size
            num_workers: Number of data loading workers
            train_transforms: Transforms for training
            val_transforms: Transforms for validation
            base_h: Base height for resizing
            base_shape: List of [width, height] shapes
            padding: Whether to pad images
            max_text_length: Maximum text length
        """
        super().__init__()
        self.train_data_dir = train_data_dir
        self.train_jsonl_paths = train_jsonl_paths
        self.val_data_dir = val_data_dir
        self.val_jsonl_paths = val_jsonl_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms or []
        self.val_transforms = val_transforms or []
        self.base_h = base_h
        self.base_shape = base_shape
        self.padding = padding
        self.max_text_length = max_text_length
        self.character_dict_path = character_dict_path
        self.use_space_char = use_space_char
        
        self.train_dataset = None
        self.val_dataset = None

    @classmethod
    def from_config(cls, config):
        """
        Create DataModule from config dict
        
        Args:
            config: Configuration dict with Train and Eval sections
            
        Returns:
            SVTRv2DataModule instance
        """
        global_config = config.get("Global", {})
        train_config = config.get("Train", {})
        eval_config = config.get("Eval", {})
        
        train_dataset_config = train_config.get("dataset", {})
        train_loader_config = train_config.get("loader", {})
        eval_dataset_config = eval_config.get("dataset", {}) if eval_config else {}
        
        # Parse train data configuration
        train_data_dir_list = train_dataset_config.get("data_dir_list", [])
        train_jsonl_file = train_dataset_config.get("jsonl_file", "train.jsonl")
        
        if isinstance(train_data_dir_list, list) and len(train_data_dir_list) > 0:
            train_data_dir = str(Path(train_data_dir_list[0]).expanduser())
            train_jsonl_path = Path(train_data_dir) / train_jsonl_file
            if not train_jsonl_path.exists():
                train_jsonl_path = Path(train_data_dir) / "dataset.jsonl"
            
            train_jsonl_paths = None
            if len(train_data_dir_list) > 1:
                train_jsonl_paths = []
                for d in train_data_dir_list:
                    d_path = Path(d).expanduser()
                    jsonl_path = d_path / train_jsonl_file
                    if not jsonl_path.exists():
                        jsonl_path = d_path / "dataset.jsonl"
                    if jsonl_path.exists():
                        train_jsonl_paths.append(str(jsonl_path))
                train_jsonl_paths = train_jsonl_paths if train_jsonl_paths else None
            else:
                train_jsonl_paths = [str(train_jsonl_path)] if train_jsonl_path.exists() else None
        else:
            train_data_dir = str(Path(train_dataset_config.get("data_dir", ".")).expanduser())
            train_jsonl_paths = None
        
        # Parse eval data configuration
        val_data_dir = None
        val_jsonl_paths = None
        eval_data_dir_list = eval_dataset_config.get("data_dir_list", [])
        val_jsonl_file = eval_dataset_config.get("jsonl_file", "val.jsonl")
        
        if isinstance(eval_data_dir_list, list) and len(eval_data_dir_list) > 0:
            val_data_dir = str(Path(eval_data_dir_list[0]).expanduser())
            val_jsonl_path = Path(val_data_dir) / val_jsonl_file
            if not val_jsonl_path.exists():
                val_jsonl_path = Path(val_data_dir) / "dataset.jsonl"
            
            if len(eval_data_dir_list) > 1:
                val_jsonl_paths = []
                for d in eval_data_dir_list:
                    d_path = Path(d).expanduser()
                    jsonl_path = d_path / val_jsonl_file
                    if not jsonl_path.exists():
                        jsonl_path = d_path / "dataset.jsonl"
                    if jsonl_path.exists():
                        val_jsonl_paths.append(str(jsonl_path))
                val_jsonl_paths = val_jsonl_paths if val_jsonl_paths else None
            else:
                val_jsonl_paths = [str(val_jsonl_path)] if val_jsonl_path.exists() else None
        
        return cls(
            train_data_dir=train_data_dir,
            train_jsonl_paths=train_jsonl_paths,
            val_data_dir=val_data_dir,
            val_jsonl_paths=val_jsonl_paths,
            batch_size=train_loader_config.get("batch_size_per_card", 32),
            num_workers=train_loader_config.get("num_workers", 4),
            max_text_length=global_config.get("max_text_length", 25),
            character_dict_path=global_config.get("character_dict_path"),
            use_space_char=global_config.get("use_space_char", False),
        )

    def setup(self, stage=None):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = SVTRv2Dataset(
                data_dir=self.train_data_dir,
                jsonl_paths=self.train_jsonl_paths,
                transforms=self.train_transforms,
                base_h=self.base_h,
                base_shape=self.base_shape,
                padding=self.padding,
                max_text_length=self.max_text_length,
                character_dict_path=self.character_dict_path,
                use_space_char=self.use_space_char,
            )
            
            if self.val_data_dir or self.val_jsonl_paths:
                self.val_dataset = SVTRv2Dataset(
                    data_dir=self.val_data_dir or self.train_data_dir,
                    jsonl_paths=self.val_jsonl_paths,
                    transforms=self.val_transforms,
                    base_h=self.base_h,
                    base_shape=self.base_shape,
                    padding=False,  # No padding for validation
                    max_text_length=self.max_text_length,
                    character_dict_path=self.character_dict_path,
                    use_space_char=self.use_space_char,
                )

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

