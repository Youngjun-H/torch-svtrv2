"""Lightning Module for SVTRv2"""
import lightning as L
import torch
import torch.nn as nn

from ..losses import build_loss
from ..metrics import build_metric
from ..postprocess import build_post_process


class LitSVTRv2(L.LightningModule):
    """Lightning Module for SVTRv2 training"""

    def __init__(self, model, config):
        """
        Args:
            model: SVTRv2Model instance
            config: Configuration dict
        """
        super().__init__()
        self.model = model
        self.config = config
        
        # Build loss
        loss_config = config.get("Loss", {})
        self.loss_fn = build_loss(loss_config)
        
        # Build metric
        metric_config = config.get("Metric", {})
        self.metric = build_metric(metric_config)
        
        # Build post-process
        postprocess_config = config.get("PostProcess", {})
        global_config = config.get("Global", {})
        self.postprocess = build_post_process(postprocess_config, global_config)
        
        # Training config
        self.train_config = config.get("Train", {})
        self.optimizer_config = config.get("Optimizer", {})
        self.scheduler_config = config.get("LRScheduler", {})

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        images = batch["image"]
        label_indices = batch["label_indices"]  # Already encoded
        label_lengths = batch["label_length"]  # Tensor
        
        # Forward pass
        preds = self.model(images)
        
        # Calculate loss
        loss_batch = {
            "label_indices": label_indices,
            "label_length": label_lengths,
        }
        loss_dict = self.loss_fn(preds, loss_batch)
        loss = loss_dict["loss"]
        
        # Log loss
        batch_size = images.size(0)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        
        return loss

    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Lightning automatically handles epoch-level logging
        # This method is here for any custom epoch-end logic if needed
        pass

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images = batch["image"]
        labels = batch["label"]  # Original strings for metrics
        label_indices = batch["label_indices"]  # Already encoded
        label_lengths = batch["label_length"]
        
        # Forward pass
        preds = self.model(images)
        
        # Calculate loss
        loss_batch = {
            "label_indices": label_indices,
            "label_length": label_lengths,
        }
        loss_dict = self.loss_fn(preds, loss_batch)
        loss = loss_dict["loss"]
        
        # Post-process predictions
        pred_texts = self.postprocess(preds, batch=None, torch_tensor=True)
        
        # Prepare labels for metric
        label_list = [(label, 1.0) for label in labels]
        
        # Update metric
        self.metric((pred_texts, label_list))
        
        # Log metrics (don't show in progress bar during step, only at epoch end)
        batch_size = images.size(0)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        
        return {"loss": loss, "preds": pred_texts, "labels": labels}

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Get metric values
        metric_dict = self.metric.get_metric(training=False)
        
        # Log metrics
        for key, value in metric_dict.items():
            # Only show main indicator in progress bar
            is_main = (key == self.metric.main_indicator)
            self.log(
                f"val/{key}",
                value,
                prog_bar=is_main,
                logger=True,
                sync_dist=True,
                batch_size=1,  # Epoch-level metrics don't have batch_size
            )
        
        # Reset metric
        self.metric.reset()

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Optimizer
        optimizer_name = self.optimizer_config.get("name", "AdamW")
        lr = self.optimizer_config.get("lr", 1e-3)
        weight_decay = self.optimizer_config.get("weight_decay", 0.0)
        
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Scheduler
        scheduler_config = self.scheduler_config.copy()
        scheduler_name = scheduler_config.pop("name", None)
        
        if scheduler_name == "OneCycleLR":
            max_epochs = self.train_config.get("max_epochs", 20)
            steps_per_epoch = self.train_config.get("steps_per_epoch", 1000)
            warmup_epoch = scheduler_config.pop("warmup_epoch", 0)
            pct_start = warmup_epoch / max_epochs if max_epochs > 0 else 0.1
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=max_epochs * steps_per_epoch,
                pct_start=pct_start,
                **scheduler_config,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif scheduler_name is None:
            return optimizer
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")


