from losses.base_loss import BaseLoss
import torch.nn.functional as F

class CustomLoss(BaseLoss):
    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs, targets)
        return loss