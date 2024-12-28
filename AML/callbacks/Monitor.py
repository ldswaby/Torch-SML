import warnings
from typing import Optional

import torch

from AML.callbacks import Callback, CALLBACK_REGISTRY


@CALLBACK_REGISTRY.register()
class Monitor(Callback):
    """Abstract class for tracking a particular metric"""

    monitor_options = ['train_loss', 'train_acc' 'val_loss', 'val_acc']
    mode_options = ['min', 'max']

    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'min',
        delta: float = 0.0,
        verbose: bool = False
    ) -> None:

        for value, options in zip(
            [monitor, mode], [self.monitor_options, self.mode_options]
        ):
            if value not in options:
                warnings.warn(
                    f"Unexpected value for Earlystopping arg: '{value}' (should"
                    f" be one of {', '.join(options)}), fallback to minimizing val_loss"
                )
                self._set_defaults()
                break
        else:
            self.monitor = monitor
            self.mode = mode

        self.delta = abs(delta)
        self.verbose = verbose
        return

    def reset(self):
        """Initialize here so subclass instance can be re-used"""
        self.best_epoch = 0
        self.monitor_op = torch.less if self.mode == 'min' else torch.greater

        if self.monitor_op == torch.less:
            self.delta = -self.delta
            self.best_score = torch.inf
        else:
            self.delta = self.delta
            self.best_score = -torch.inf
        return

    def _set_defaults(self):
        """_summary_
        """
        self.monitor = 'val_loss'
        self.mode = 'min'

    def get_monitor_value(self, logs: Optional[dict] = None):
        logs = logs or {}
        return logs.get(self.monitor)

    def is_improvement(self, monitor_value, reference_value):
        """_summary_

        Args:
            monitor_value (_type_): _description_
            reference_value (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.monitor_op(monitor_value - self.delta, reference_value)
