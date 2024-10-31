from typing import List, Optional, Union

from AML.metrics import Metric, MetricCollection


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
