"""Encoders package"""
from .svtrv2_lnconv_two33 import SVTRv2LNConvTwo33

__all__ = ["SVTRv2LNConvTwo33", "build_encoder"]

# Encoder name to class mapping
ENCODER_CLASSES = {
    "SVTRv2LNConvTwo33": SVTRv2LNConvTwo33,
}


def build_encoder(config):
    """Build encoder from config"""
    config = config.copy()
    module_name = config.pop("name")
    
    if module_name not in ENCODER_CLASSES:
        raise ValueError(
            f"Unsupported encoder: {module_name}. "
            f"Supported encoders: {list(ENCODER_CLASSES.keys())}"
        )
    
    encoder_class = ENCODER_CLASSES[module_name]
    return encoder_class(**config)

