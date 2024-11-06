from ..utils.registry import Registry
from .base import BaseDataset
from .csv_dataset import CSVDataset
# from .hdf5_dataset import HDF5Dataset
# from .image_dataset import ImageDataset
from .numpy_dataset import NumpyDataset
from .pandas_dataset import PandasDataset

DATASET_REGISTRY = Registry('Dataset')
