from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class DataSplitter(ABC):
    """
    Abstract base class for different dataset splitting strategies.

    Methods:
        split(dataset): Splits the given dataset and returns one or more subsets.
    """

    @abstractmethod
    def split(self, dataset: Dataset) -> Any:
        """
        Splits the given dataset according to the splitting strategy.

        Args:
            dataset (Dataset): The dataset to split.

        Returns:
            Any: The subsets or data structures containing the split datasets.
        """