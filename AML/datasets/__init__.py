from typing import Dict
from torch.utils.data import DataLoader

from ..utils.registry import Registry
DATASET_REGISTRY = Registry('Dataset')

from .base import BaseDataset
from .csv_dataset import CSVDataset
# from .hdf5_dataset import HDF5Dataset
# from .image_dataset import ImageDataset
from .numpy_dataset import NumpyDataset
from .pandas_dataset import PandasDataset
from . import vision


def _build_dataset(config: dict) -> BaseDataset:
    """Returns CompositeLoss object

    Args:
        config (dict): _description_

    Returns:
        dict: _description_
    """
    return DATASET_REGISTRY.get(config['DATASET']['name'])(
        **config['DATASET']['kwargs']
    )


def _build_dataloaders(config: dict, dataset: BaseDataset) -> Dict[str, DataLoader]:
    """Builds train/validation/test dataloaders

    Args:
        config (dict): _description_
        dataset (BaseDataset): _description_

    Returns:
        DataLoader: _description_
    """
    # TODO
    pass
