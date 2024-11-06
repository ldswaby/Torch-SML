from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class ensuring __getitem__ outputs are dictionaries
    with 'data' and 'target' keys.

    The data and target are expected to be PyTorch tensors.
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve a sample and its target by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary with 'data' and 'target' keys,
            both of which are PyTorch tensors.
        """
        data = self.get_data(idx)
        target = self.get_target(idx)
        return {'data': data, 'target': target}

    @abstractmethod
    def get_data(self, idx: int) -> torch.Tensor:
        """Retrieve the data sample by index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            torch.Tensor: The data sample as a PyTorch tensor.
        """

    @abstractmethod
    def get_target(self, idx: int) -> torch.Tensor:
        """Retrieve the target by index.

        Args:
            idx (int): Index of the target.

        Returns:
            torch.Tensor: The target corresponding to the data sample as a PyTorch tensor.
        """

    @property
    def in_shape(self):
        return self[0]['data'].shape

    @property
    def num_classes(self):
        return len(set(x['target'].item() for x in self))
