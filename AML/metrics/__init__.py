import importlib

import torchmetrics
from torchmetrics import Metric, MetricCollection

from .metrics import *

# Load the adjacent metrics.py as a module to use with getattr
metrics_module = importlib.import_module(".metrics", package=__name__)


def _fetch_metric(metric_name: str):
    """
    Retrieves a function or class by name from either torchmetrics or the adjacent metrics.py file.

    Args:
        metric_name (str): The name of the function or class to retrieve.

    Returns:
        The function or class if found, otherwise raises AttributeError.
    """
    # First, try to fetch from the adjacent metrics.py file
    if hasattr(metrics_module, metric_name):
        return getattr(metrics_module, metric_name)

    # If not found in metrics.py, try to fetch from torchmetrics
    if hasattr(torchmetrics, metric_name):
        return getattr(torchmetrics, metric_name)

    # Raise an error if the metric was not found in either location
    raise AttributeError(
        f"'{metric_name}' not found in metrics.py or torchmetrics")


def build_metrics(config: dict) -> dict:
    """Returns dict of MetricCollectios (one per dataset)

    Args:
        config (dict): _description_

    Returns:
        dict: _description_
    """
    metrics = {}

    for dset, _mtrcs in config['METRICS'].items():
        dset_mtrcs = []
        for name, kwargs in _mtrcs.items():
            dset_mtrcs.append(_fetch_metric(name)(**kwargs))
        metrics[dset] = MetricCollection(dset_mtrcs, prefix=f'{dset}/')

    return metrics
