import h5py
import torch

from AML.datasets import DATASET_REGISTRY, BaseDataset


@DATASET_REGISTRY.register('HDF5Dataset')
class HDF5Dataset(BaseDataset):
    """Dataset for loading data and targets from an HDF5 file.

    Args:
        hdf5_file (str): Path to the HDF5 file.
        data_key (str, optional): Key for the data in the HDF5 file.
            Defaults to 'data'.
        target_key (str, optional): Key for the targets in the HDF5 file.
            Defaults to 'target'.
    """

    def __init__(
        self,
        hdf5_file: str,
        data_key: str = 'data',
        target_key: str = 'target'
    ) -> None:
        self.hdf5_file = hdf5_file
        self.data_key = data_key
        self.target_key = target_key
        self.file = h5py.File(hdf5_file, 'r')
        self.data = self.file[self.data_key]
        self.targets = self.file[self.target_key]

    def __len__(self) -> int:
        return len(self.data)

    def get_data(self, idx: int) -> torch.Tensor:
        data = self.data[idx][()]
        data = torch.from_numpy(data)
        return data

    def get_target(self, idx: int) -> torch.Tensor:
        target = self.targets[idx][()]
        target = torch.tensor(target)
        return target

    def __del__(self) -> None:
        if hasattr(self, 'file') and self.file:
            self.file.close()
