import os

import numpy as np
import torch

from AML.datasets import DATASET_REGISTRY, BaseDataset


@DATASET_REGISTRY.register('NumpyDataset')
class NumpyDataset(BaseDataset):
    """Dataset for loading data and targets from NumPy files.

    Args:
        root (str): Root directory containing the NumPy files.
        data_file (str, optional): Filename of the NumPy file containing data.
            Defaults to 'data.npy'.
        target_file (str, optional): Filename of the NumPy file containing targets.
            Defaults to 'target.npy'.
    """

    def __init__(
        self,
        root: str,
        data_file: str = 'data.npy',
        target_file: str = 'target.npy'
    ) -> None:
        self.data = np.load(os.path.join(root, data_file))
        self.targets = np.load(os.path.join(root, target_file))
        assert len(self.data) == len(self.targets), \
            "Data and target lengths do not match."

    def __len__(self) -> int:
        return len(self.data)

    def get_data(self, idx: int) -> torch.Tensor:
        data = self.data[idx]
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        return data

    def get_target(self, idx: int) -> torch.Tensor:
        target = self.targets[idx]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
        return target
