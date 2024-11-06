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
