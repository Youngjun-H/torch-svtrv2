"""Loss functions"""
from .ctc_loss import CTCLoss

__all__ = ["CTCLoss", "build_loss"]

LOSS_CLASSES = {
    "CTCLoss": CTCLoss,
}


def build_loss(config):
    """Build loss from config"""
    config = config.copy()
    loss_name = config.pop("name")
    
    if loss_name not in LOSS_CLASSES:
        raise ValueError(
            f"Unsupported loss: {loss_name}. "
            f"Supported losses: {list(LOSS_CLASSES.keys())}"
        )
    
    loss_class = LOSS_CLASSES[loss_name]
    return loss_class(**config)

