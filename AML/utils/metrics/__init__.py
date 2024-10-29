# from .base_metric import Metric
# from .metrics import *
import torchmetrics
from torchmetrics import MetricCollection


def build_metrics(cfg: dict):
    """Returnr MetricCollection object

    Args:
        cfg (dict): _description_

    Returns:
        _type_: _description_
    """
    return MetricCollection([
        getattr(torchmetrics, name)(**kwargs) for name, kwargs in cfg['METRICS'].items()
    ])
