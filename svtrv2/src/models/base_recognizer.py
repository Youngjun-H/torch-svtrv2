"""Base Recognizer for SVTRv2"""
import torch
from torch import nn

from .decoders import build_decoder
from .encoders import build_encoder


class BaseRecognizer(nn.Module):
    """Base Recognizer model"""

    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary with Encoder and Decoder configs
        """
        super(BaseRecognizer, self).__init__()
        in_channels = config.get("in_channels", 3)
        self.use_wd = config.get("use_wd", True)

        # Transform is not used in SVTRv2
        self.use_transform = False

        # Build encoder
        if "Encoder" not in config or config["Encoder"] is None:
            self.use_encoder = False
        else:
            self.use_encoder = True
            encoder_config = config["Encoder"].copy()
            encoder_config["in_channels"] = in_channels
            self.encoder = build_encoder(encoder_config)
            in_channels = self.encoder.out_channels

        # Build decoder
        if "Decoder" not in config or config["Decoder"] is None:
            self.use_decoder = False
        else:
            self.use_decoder = True
            decoder_config = config["Decoder"].copy()
            decoder_config["in_channels"] = in_channels
            self.decoder = build_decoder(decoder_config)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return parameters that should not have weight decay"""
        if self.use_wd:
            no_weight_decay = {}
            if self.use_encoder and hasattr(self.encoder, "no_weight_decay"):
                no_weight_decay.update(self.encoder.no_weight_decay())
            if self.use_decoder and hasattr(self.decoder, "no_weight_decay"):
                no_weight_decay.update(self.decoder.no_weight_decay())
            return no_weight_decay
        else:
            return {}

    def forward(self, x, data=None):
        """Forward pass"""
        if self.use_transform:
            x = self.transform(x)
        if self.use_encoder:
            x = self.encoder(x)
        if self.use_decoder:
            x = self.decoder(x, data=data)
        return x

