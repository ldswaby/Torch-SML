import warnings
from typing import Optional, Union, List

import torchmetrics
from torchmetrics import Metric, MetricCollection
from torchmetrics import classification, regression, detection, functional, image

from ..utils.registry import Registry
METRIC_REGISTRY = Registry('Metric')

from . import utils
from .metrics import *

modules = [
    torchmetrics,
    classification,
    regression,
    detection,
    functional,
    image
]

# Register torchmetrics
for module in modules:
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        # Check if the attribute is a class and a subclass of torchmetrics.Metric
        if isinstance(attr, type) and issubclass(attr, Metric) and attr is not Metric:
            # Register the metric with its class name
            try:
                globals()[attr_name] = attr
                METRIC_REGISTRY.register(name=attr_name)(attr)
            except KeyError:
                # Metric already registered, skip
                continue


del module, modules, attr, attr_name


def _build_metrics(config: dict) -> dict:
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
            dset_mtrcs.append(METRIC_REGISTRY.get(name)(**kwargs))
        metrics[dset] = MetricCollection(dset_mtrcs, prefix=f'{dset}/')

    return metrics


def _process_metrics(metrics: Optional[Union[List[Metric], MetricCollection]] = None):
    """Ensures all metrics are in MetricCollection object

    Args:
        metrics (Optional[Union[List[Metric], MetricCollection]], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    metrics = metrics or []
    if not isinstance(metrics, MetricCollection):
        metrics = MetricCollection(metrics)
    return metrics

# Namespace cleanup
del warnings
del torchmetrics
del Union, Optional, List