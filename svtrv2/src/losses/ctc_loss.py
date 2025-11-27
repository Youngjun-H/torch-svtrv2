"""CTC Loss"""
import torch
from torch import nn


class CTCLoss(nn.Module):
    """CTC Loss"""

    def __init__(self, use_focal_loss=False, zero_infinity=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(
            blank=0, reduction="none", zero_infinity=zero_infinity
        )
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        """
        Args:
            predicts: Model predictions [B, W, C]
            batch: Tuple of (images, labels, label_lengths) or dict with 'label' and 'label_length'
        """
        batch_size = predicts.size(0)
        
        # Handle both tuple and dict batch formats
        if isinstance(batch, dict):
            label = batch.get("label_indices", batch.get("label"))  # Prefer label_indices
            label_length = batch["label_length"]
        else:
            label, label_length = batch[1], batch[2]
        
        predicts = predicts.log_softmax(2)
        predicts = predicts.permute(1, 0, 2)  # [W, B, C]
        preds_lengths = torch.tensor(
            [predicts.size(0)] * batch_size, dtype=torch.long, device=predicts.device
        )
        
        loss = self.loss_func(predicts, label, preds_lengths, label_length)

        if self.use_focal_loss:
            clamped_loss = torch.clamp(loss, min=-20, max=20)
            weight = 1 - torch.exp(-clamped_loss)
            weight = torch.square(weight)
            loss = torch.where(weight > 0, loss * weight, loss)
        
        loss = loss.mean()
        return {"loss": loss}

