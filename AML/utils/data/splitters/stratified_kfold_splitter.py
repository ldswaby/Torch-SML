from typing import Any, Dict, List

from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from sklearn.model_selection import StratifiedKFold

from AML.utils.data.splitters import DataSplitter, DATA_SPLITTER_REGISTRY


@DATA_SPLITTER_REGISTRY.register('stratified_kfold')
class StratifiedKFoldSplitter(DataSplitter):
    """
    Performs Stratified K-Fold Cross-Validation splitting for classification tasks.

    Attributes:
        n_splits (int): Number of folds.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_seed (int): Random seed for reproducibility.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_seed: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self, dataset: Dataset, targets: List[Any]) -> List[Dict[str, Subset]]:
        """
        Splits the dataset into K stratified folds for cross-validation.

        Args:
            dataset (Dataset): The dataset to split.
            targets (List[Any]): List of labels/targets corresponding to the dataset.

        Returns:
            List[Dict[str, Subset]]: A list of dictionaries, each containing
                'train' and 'val' subsets for each fold.
        """
        indices = list(range(len(dataset)))
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_seed
        )

        folds = []
        for train_idx, val_idx in skf.split(indices, targets):
            train_data = Subset(dataset, train_idx)
            val_data = Subset(dataset, val_idx)
            folds.append({
                "train": train_data,
                "val": val_data
            })
        return folds
