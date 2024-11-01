import torch
from torch import nn
from typing import List, Dict, Optional


class CompositeLoss(nn.Module):
    """
    A module to combine multiple loss functions into a single loss, while returning
    individual losses as a dictionary.

    Args:
        losses (List[nn.Module]):
            A list of loss function instances.
        weights (List[float], optional):
            A list of weights corresponding to each loss function.
            If not provided, all losses are weighted equally.
    """

    def __init__(
        self,
        losses: List[nn.Module],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        if weights is None:
            # Assign equal weight to all losses
            self.weights = [1.0] * len(self.losses)
        else:
            if len(weights) != len(losses):
                raise ValueError(
                    "The number of weights must match the number of losses."
                )
            self.weights = weights

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss and return individual losses.

        Args:
            outputs (torch.Tensor): The model outputs.
            targets (torch.Tensor): The ground truth targets.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing individual losses and the total loss.
        """
        losses_dict = {}
        total_loss = torch.tensor(0.0, device=outputs.device)
        for loss_fn, weight in zip(self.losses, self.weights):
            loss_value = loss_fn(outputs, targets) * weight
            loss_name = loss_fn.__class__.__name__
            losses_dict[loss_name] = loss_value
            total_loss += loss_value
        losses_dict['loss_total'] = total_loss
        return losses_dict