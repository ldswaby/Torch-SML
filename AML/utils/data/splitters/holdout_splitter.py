from typing import Dict

import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataset import Subset

from AML.utils.data.splitters import DataSplitter, DATA_SPLITTER_REGISTRY


@DATA_SPLITTER_REGISTRY.register('holdout')
class HoldoutSplitter(DataSplitter):
    """
    Randomly splits the dataset into train, validation, and test splits (holdout
    method).

    Attributes:
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_seed: int = 42,
    ):
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"Sum of train, val, and test ratios must be 1.0. Got {train_ratio + val_ratio + test_ratio:.3f}."
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self, dataset: Dataset) -> Dict[str, Subset]:
        """
        Splits the dataset into train, validation, and test sets using the holdout method.

        Args:
            dataset (Dataset): The dataset to split.

        Returns:
            Dict[str, Subset]: A dictionary with keys 'train', 'val', and 'test',
                mapped to their respective Subset objects.
        """
        dataset_size = len(dataset)
        train_len = int(self.train_ratio * dataset_size)
        val_len = int(self.val_ratio * dataset_size)
        test_len = dataset_size - train_len - val_len

        # Using PyTorch's random_split to create subsets
        train_set, val_set, test_set = random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.random_seed) if self.shuffle else None
        )

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }