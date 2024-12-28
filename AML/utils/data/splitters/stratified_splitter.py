from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split

from AML.utils.data.splitters import DataSplitter, DATA_SPLITTER_REGISTRY


@DATA_SPLITTER_REGISTRY.register('stratified')
class StratifiedSplitter(DataSplitter):
    """
    Splits a classification dataset into train, validation, and test splits (stratified holdout).

    Attributes:
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        random_seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"Sum of train, val, and test ratios must be 1.0. Got {train_ratio + val_ratio + test_ratio:.3f}."
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split(
        self,
        dataset: Dataset,
        targets: List[Any]
    ) -> Dict[str, Dataset]:
        """
        Stratified split for classification tasks. Requires access to labels/targets.

        Args:
            dataset (Dataset): The dataset to split.
            targets (List[Any]): List of labels/targets corresponding to the dataset.

        Returns:
            Dict[str, Dataset]: Subsets with keys 'train', 'val', 'test'.
        """
        if len(dataset) != len(targets):
            raise ValueError("Dataset length and targets length do not match.")

        # First split (train + val / test)
        X_temp_indices, X_test_indices, y_temp, y_test = train_test_split(
            range(len(dataset)),
            targets,
            test_size=self.test_ratio,
            stratify=targets,
            random_state=self.random_seed
        )

        # Second split (train / val)
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train_indices, X_val_indices, y_train, y_val = train_test_split(
            X_temp_indices,
            y_temp,
            test_size=val_ratio_adjusted,
            stratify=y_temp,
            random_state=self.random_seed
        )

        # Convert indices to Subset objects
        train_data = Subset(dataset, X_train_indices)
        val_data = Subset(dataset, X_val_indices)
        test_data = Subset(dataset, X_test_indices)

        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }