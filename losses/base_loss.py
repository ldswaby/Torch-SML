import torch.nn as nn

class BaseLoss(nn.Module):
    def __init__(self, config):
        super(BaseLoss, self).__init__()
        self.config = config

    def forward(self, outputs, targets):
        raise NotImplementedError("Subclasses should implement this method.")