from AML.loss import Module

import torch.nn.functional as F

class CustomLoss(Module):
    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs, targets)
        return loss