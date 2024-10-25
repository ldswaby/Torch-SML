from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = []
        self.targets = []
        self.load_data()

    def load_data(self):
        """Load and preprocess data."""
        raise NotImplementedError("Subclasses should implement this method.")

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


import torch
from data.datasets import BaseDataset

class CustomDataset(BaseDataset):
    def load_data(self):
        # Load your data here
        self.data = torch.randn(1000, self.config['input_size'])
        self.targets = torch.randint(0, self.config['output_size'], (1000,))