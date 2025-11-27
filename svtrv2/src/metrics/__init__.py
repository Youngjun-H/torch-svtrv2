"""Metrics"""
from .rec_metric import RecMetric

__all__ = ["RecMetric", "build_metric"]

METRIC_CLASSES = {
    "RecMetric": RecMetric,
}


def build_metric(config):
    """Build metric from config"""
    config = config.copy()
    metric_name = config.pop("name")
    
    if metric_name not in METRIC_CLASSES:
        raise ValueError(
            f"Unsupported metric: {metric_name}. "
            f"Supported metrics: {list(METRIC_CLASSES.keys())}"
        )
    
    metric_class = METRIC_CLASSES[metric_name]
    return metric_class(**config)

