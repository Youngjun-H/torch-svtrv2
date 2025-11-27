"""Lightning DataModule for SVTRv2"""
import lightning as L
from torch.utils.data import DataLoader

from svtrv2.src.data.dataset import SVTRv2Dataset


class SVTRv2DataModule(L.LightningDataModule):
    """Simple DataModule for SVTRv2 training"""

    def __init__(
        self,
        train_data_dir,
        val_data_dir=None,
        batch_size=32,
        num_workers=4,
        train_transforms=None,
        val_transforms=None,
        base_h=32,
        base_shape=None,
        padding=True,
        max_text_length=25,
    ):
        """
        Args:
            train_data_dir: Directory containing training data (with dataset.jsonl)
            val_data_dir: Directory containing validation data (optional)
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
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms or []
        self.val_transforms = val_transforms or []
        self.base_h = base_h
        self.base_shape = base_shape
        self.padding = padding
        self.max_text_length = max_text_length
        
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = SVTRv2Dataset(
                data_dir=self.train_data_dir,
                transforms=self.train_transforms,
                base_h=self.base_h,
                base_shape=self.base_shape,
                padding=self.padding,
                max_text_length=self.max_text_length,
            )
            
            if self.val_data_dir:
                self.val_dataset = SVTRv2Dataset(
                    data_dir=self.val_data_dir,
                    transforms=self.val_transforms,
                    base_h=self.base_h,
                    base_shape=self.base_shape,
                    padding=False,  # No padding for validation
                    max_text_length=self.max_text_length,
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
        )

