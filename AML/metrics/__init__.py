import warnings, inspect
from typing import List, Optional, Union

import torchmetrics
from torchmetrics import (Metric, MetricCollection, classification, detection,
                          functional, image, regression)

from ..utils.registry import Registry
METRIC_REGISTRY = Registry('Metric')

from . import metrics


modules = [
    torchmetrics,
    classification,
    regression,
    detection,
    functional,
    image,
    metrics  # Register any additional custom metrics modules here
]

# Register torchmetrics
for module in modules:
    for name, obj in inspect.getmembers(module):
        # Check if the attribute is a class and a subclass of torchmetrics.Metric
        if inspect.isclass(obj) and issubclass(obj, Metric) and obj is not Metric:
            # Register the metric with its class name
            try:
                globals()[name] = obj
                METRIC_REGISTRY.register(name=name)(obj)
            except KeyError:
                # Metric already registered, skip
                continue

del module, modules, obj, name

def _build_metrics(config: dict) -> dict:
    """Returns dict of MetricCollectios (one per dataset)

    Args:
        config (dict): _description_

    Returns:
        dict: _description_
    """
    metrics_dict = {}

    for dset, _mtrcs in config['METRICS'].items():
        dset = dset.lower()
        dset_mtrcs = []
        for _m in _mtrcs:
            dset_mtrcs.append(METRIC_REGISTRY.get(_m['name'])(**_m['kwargs']))
        metrics_dict[dset] = MetricCollection(dset_mtrcs, prefix=f'{dset}/')

    return metrics_dict


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
