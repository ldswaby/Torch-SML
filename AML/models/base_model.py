import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        """Define the model layers."""
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x):
        """Forward pass logic."""
        raise NotImplementedError("Subclasses should implement this method.")