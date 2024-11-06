from typing import List

import pandas as pd
import torch

from AML.datasets import BaseDataset


class CSVDataset(BaseDataset):
    """Dataset for loading data and targets from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        input_columns (List[str]): List of column names to be used as input data.
        target_column (str): Column name to be used as target.
    """

    def __init__(
        self,
        csv_file: str,
        input_columns: List[str],
        target_column: str
    ) -> None:
        self.df = pd.read_csv(csv_file)
        self.input_columns = input_columns
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.df)

    def get_data(self, idx: int) -> torch.Tensor:
        data = self.df.iloc[idx][self.input_columns].values.astype('float32')
        data = torch.from_numpy(data)
        return data

    def get_target(self, idx: int) -> torch.Tensor:
        target = self.df.iloc[idx][self.target_column]
        target = torch.tensor(target)
        return target
