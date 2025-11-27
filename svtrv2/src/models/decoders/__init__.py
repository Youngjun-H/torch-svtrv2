"""Decoders package"""
from .rctc_decoder import RCTCDecoder

__all__ = ["RCTCDecoder", "build_decoder"]

# Decoder name to class mapping
DECODER_CLASSES = {
    "RCTCDecoder": RCTCDecoder,
}


def build_decoder(config):
    """Build decoder from config"""
    config = config.copy()
    module_name = config.pop("name")
    
    if module_name not in DECODER_CLASSES:
        raise ValueError(
            f"Unsupported decoder: {module_name}. "
            f"Supported decoders: {list(DECODER_CLASSES.keys())}"
        )
    
    decoder_class = DECODER_CLASSES[module_name]
    return decoder_class(**config)

