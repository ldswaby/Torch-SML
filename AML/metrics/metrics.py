"""
To implement a custom metric, see:
https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
"""

import torch
from torch import Tensor

from AML.metrics import Metric, METRIC_REGISTRY

__all__ = [
    'CustomAccuracy'
    # ...
]


@METRIC_REGISTRY.register('CustomAccuracy')
class CustomAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._input_format(preds, target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
