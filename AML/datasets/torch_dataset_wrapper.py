from torch.utils.data import Dataset

class TorchDatasetWrapper(Dataset):
    """
    A wrapper for PyTorch classification datasets to return samples as dictionaries
    with keys 'data' and 'target'.
    """
    def __init__(self, dataset: Dataset):
        """
        Args:
            dataset (Dataset): A PyTorch dataset instance to be wrapped.
        """
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the sample from the underlying dataset
        sample = self.dataset[idx]

        # Check if the dataset already returns (data, target) as a tuple
        if isinstance(sample, tuple) and len(sample) == 2:
            data, target = sample
            return {'data': data, 'target': target}

        # Raise an error for unsupported datasets
        raise ValueError(
            "The provided dataset must return samples as (data, target) tuples."
        )
