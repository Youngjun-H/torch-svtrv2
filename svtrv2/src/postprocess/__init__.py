"""Post-processing"""
from .ctc_postprocess import CTCLabelDecode

__all__ = ["CTCLabelDecode", "build_post_process"]

POSTPROCESS_CLASSES = {
    "CTCLabelDecode": CTCLabelDecode,
}


def build_post_process(config, global_config=None):
    """Build post-process from config"""
    config = config.copy()
    if global_config is not None:
        config.update(global_config)
    
    postprocess_name = config.pop("name")
    
    if postprocess_name not in POSTPROCESS_CLASSES:
        raise ValueError(
            f"Unsupported postprocess: {postprocess_name}. "
            f"Supported postprocesses: {list(POSTPROCESS_CLASSES.keys())}"
        )
    
    postprocess_class = POSTPROCESS_CLASSES[postprocess_name]
    return postprocess_class(**config)

