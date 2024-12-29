import inspect
from typing import Dict
from torch.utils.data import DataLoader, Dataset

from ..utils.registry import Registry
DATASET_REGISTRY = Registry('Dataset')

from ..utils.data.splitters import _data_splitter_factory

from .base import BaseDataset
from .csv_dataset import CSVDataset
# from .hdf5_dataset import HDF5Dataset
# from .image_dataset import ImageDataset
from .numpy_dataset import NumpyDataset
from .pandas_dataset import PandasDataset
from . import vision
from .torch_dataset_wrapper import TorchDatasetWrapper


import inspect
from typing import Dict

def _build_dataset(config: dict) -> Dict[str, Dataset]:
    """
    Builds and returns a dictionary of dataset splits (e.g., train, test, val)
    based on the provided configuration. If the underlying dataset class
    supports a 'train' argument (pre-split train/test), it will load each subset
    directly. Otherwise, it will load the entire dataset and split it manually.

    Args:
        config (dict): Configuration dictionary with the following structure:
            {
                'DATASET': {
                    'name': str,             # Name of the dataset in DATASET_REGISTRY
                    'kwargs': dict           # Keyword arguments for the dataset
                },
                # Other relevant keys for splitting the dataset, e.g., ratios
            }

    Returns:
        Dict[str, BaseDataset]: A mapping from split name ('train', 'val', 'test')
        to the corresponding dataset object.
    """
    # Retrieve dataset class from the registry
    dataset_cls = DATASET_REGISTRY.get(config['DATASET']['name'])
    if dataset_cls is None:
        raise ValueError(
            f"Dataset '{config['DATASET']['name']}' not found in registry."
        )

    # Copy dataset kwargs from config
    dataset_kwargs = config['DATASET']['kwargs'].copy()

    # Create the splitter object
    splitter = _data_splitter_factory(config)

    # Check if the dataset supports a 'train' parameter in its __init__
    sig = inspect.signature(dataset_cls.__init__)

    if 'train' in sig.parameters:
        # ----------------------------------------------------------
        # 1) Dataset is already split into train and test by itself
        # ----------------------------------------------------------

        # Load the pre-split training set
        dataset_kwargs['train'] = True
        train_dataset = dataset_cls(**dataset_kwargs)

        # Load the pre-split testing set
        dataset_kwargs['train'] = False
        test_dataset = dataset_cls(**dataset_kwargs)

        datasets = {
            'train': train_dataset,
            'test': test_dataset
        }

        # If there's a validation ratio, split the train set accordingly
        if hasattr(splitter, 'val_ratio'):
            total_size = len(train_dataset) + len(test_dataset)
            num_val_samples = int(splitter.val_ratio * total_size)

            # Compute new validation ratio based on the train set size
            new_val_ratio = num_val_samples / len(train_dataset)

            # Adjust the splitter’s ratios
            splitter.test_ratio = 0
            splitter.val_ratio = new_val_ratio
            splitter.train_ratio = 1 - new_val_ratio

            # Split the train dataset into train/val
            split_result = splitter.split(train_dataset)
            datasets['train'] = split_result['train']
            datasets['val'] = split_result['val']

    else:
        # ----------------------------------------------------------
        # 2) Load the full dataset and split manually
        # ----------------------------------------------------------
        full_dataset = dataset_cls(**dataset_kwargs)
        datasets = splitter.split(full_dataset)

    # Wrap each split if it's not already a subclass of BaseDataset
    for split_name, dset in datasets.items():
        if not issubclass(type(dset), BaseDataset):
            # then it's a native torch dataset
            datasets[split_name] = TorchDatasetWrapper(dset)

    return datasets



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
