from typing import List

import pandas as pd
import torch

from AML.datasets import DATASET_REGISTRY, BaseDataset


@DATASET_REGISTRY.register('PandasDataset')
class PandasDataset(BaseDataset):
    """Dataset for loading data and targets from a pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing data.
        input_columns (List[str]): List of column names to be used as input data.
        target_column (str): Column name to be used as target.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        input_columns: List[str],
        target_column: str
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.input_columns = input_columns
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.df)

    def get_data(self, idx: int) -> torch.Tensor:
        data = self.df.loc[idx, self.input_columns].values.astype('float32')
        data = torch.from_numpy(data)
        return data

    def get_target(self, idx: int) -> torch.Tensor:
        target = self.df.loc[idx, self.target_column]
        target = torch.tensor(target)
        return target
