from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    """
    Abstract base class for all metrics.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """

    @abstractmethod
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Ground truth labels or targets.
        """

    @abstractmethod
    def compute(self) -> float:
        """
        Computes the final metric value using internal state data.

        Returns:
            float: The computed metric value.
        """

    @property
    def name(self) -> str:
        """
        Returns the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return self.__class__.__name__
