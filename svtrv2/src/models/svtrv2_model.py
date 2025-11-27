"""SVTRv2 Model"""
import copy

from .base_recognizer import BaseRecognizer


class SVTRv2Model(BaseRecognizer):
    """SVTRv2 Model wrapper"""

    def __init__(self, config):
        """
        Args:
            config: Configuration dict. Can be:
                - Full config dict with 'Architecture' key
                - Architecture config dict directly
        """
        # Handle both full config and architecture-only config
        if isinstance(config, dict) and "Architecture" in config:
            arch_config = copy.deepcopy(config["Architecture"])
        else:
            arch_config = copy.deepcopy(config)

        super().__init__(arch_config)

