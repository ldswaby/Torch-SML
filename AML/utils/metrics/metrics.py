from typing import List

import torch
import torch.nn.functional as F

from AML.utils.metrics import Metric

__all__ = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1Score',
    'ConfusionMatrix',
    'MeanSquaredError',
    'MeanAbsoluteError',
    'RootMeanSquaredError',
    'R2Score',
]


class Metrics(Metric):
    """
    Computes the accuracy classification score.
    """

    def __init__(self, metrics: List[Metric]) -> None:
        self.metrics = metrics

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        for metric in self.metrics:
            metric.reset()

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels.
        """
        with torch.no_grad():
            for metric in self.metrics:
                metric.update(outputs, targets)

    def compute(self) -> dict:
        """
        Computes the accuracy.

        Returns:
            float: The computed accuracy.
        """

        return {metric.name: metric.compute() for metric in self.metrics}


# ------------------- Classification Metrics -------------------


class Accuracy(Metric):
    """
    Computes the accuracy classification score.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        self.correct: int = 0
        self.total: int = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels.
        """
        with torch.no_grad():
            # Use torch.argmax to get predicted classes
            preds = torch.argmax(outputs, dim=1)
            # Update counts
            self.correct += (preds == targets).sum().item()
            self.total += targets.size(0)

    def compute(self) -> float:
        """
        Computes the accuracy.

        Returns:
            float: The computed accuracy.
        """
        return self.correct / self.total if self.total > 0 else 0.0


class Precision(Metric):
    """
    Computes the precision score for multiclass classification.

    Args:
        num_classes (int): Number of classes.
        average (str): Type of averaging ('macro', 'micro', or None for per-class).
    """

    def __init__(self, num_classes: int, average: str = 'macro') -> None:
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        self.true_positives = torch.zeros(self.num_classes, dtype=torch.float)
        self.predicted_positives = torch.zeros(
            self.num_classes, dtype=torch.float)

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels.
        """
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # Use F.one_hot for one-hot encoding
            preds_one_hot = F.one_hot(preds, num_classes=self.num_classes)
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)

            self.true_positives += (preds_one_hot &
                                    targets_one_hot).sum(dim=0).float()
            self.predicted_positives += preds_one_hot.sum(dim=0).float()

    def compute(self) -> float:
        """
        Computes the precision score.

        Returns:
            float: The computed precision score.
        """
        precision = self.true_positives / (self.predicted_positives + 1e-7)
        if self.average == 'macro':
            return precision.mean().item()
        elif self.average == 'micro':
            total_tp = self.true_positives.sum()
            total_pp = self.predicted_positives.sum()
            return (total_tp / (total_pp + 1e-7)).item()
        else:
            return precision  # Returns per-class precision as a tensor


class Recall(Metric):
    """
    Computes the recall score for multiclass classification.

    Args:
        num_classes (int): Number of classes.
        average (str): Type of averaging ('macro', 'micro', or None for per-class).
    """

    def __init__(self, num_classes: int, average: str = 'macro') -> None:
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        self.true_positives = torch.zeros(self.num_classes, dtype=torch.float)
        self.actual_positives = torch.zeros(
            self.num_classes, dtype=torch.float)

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels.
        """
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            preds_one_hot = F.one_hot(preds, num_classes=self.num_classes)
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)

            self.true_positives += (preds_one_hot &
                                    targets_one_hot).sum(dim=0).float()
            self.actual_positives += targets_one_hot.sum(dim=0).float()

    def compute(self) -> float:
        """
        Computes the recall score.

        Returns:
            float: The computed recall score.
        """
        recall = self.true_positives / (self.actual_positives + 1e-7)
        if self.average == 'macro':
            return recall.mean().item()
        elif self.average == 'micro':
            total_tp = self.true_positives.sum()
            total_ap = self.actual_positives.sum()
            return (total_tp / (total_ap + 1e-7)).item()
        else:
            return recall  # Returns per-class recall as a tensor


class F1Score(Metric):
    """
    Computes the F1 score for multiclass classification.

    Args:
        num_classes (int): Number of classes.
        average (str): Type of averaging ('macro', 'micro', or None for per-class).
    """

    def __init__(self, num_classes: int, average: str = 'macro') -> None:
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        self.precision_metric = Precision(
            num_classes=self.num_classes, average=None
        )
        self.recall_metric = Recall(num_classes=self.num_classes, average=None)

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels.
        """
        self.precision_metric.update(outputs, targets)
        self.recall_metric.update(outputs, targets)

    def compute(self) -> float:
        """
        Computes the F1 score.

        Returns:
            float: The computed F1 score.
        """
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        if isinstance(precision, torch.Tensor):
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            if self.average == 'macro':
                return f1.mean().item()
            elif self.average == 'micro':
                total_tp = self.precision_metric.true_positives.sum()
                total_pp = self.precision_metric.predicted_positives.sum()
                total_ap = self.recall_metric.actual_positives.sum()
                precision_micro = total_tp / (total_pp + 1e-7)
                recall_micro = total_tp / (total_ap + 1e-7)
                f1_micro = 2 * (precision_micro * recall_micro) / (
                    precision_micro + recall_micro + 1e-7
                )
                return f1_micro.item()
            else:
                return f1  # Returns per-class F1 scores as tensor
        else:
            return 2 * (precision * recall) / (precision + recall + 1e-7)


class ConfusionMatrix(Metric):
    """
    Computes the confusion matrix for multiclass classification.

    Args:
        num_classes (int): Number of classes.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        self.matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the confusion matrix with new data.

        Args:
            outputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels.
        """
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # Use torch.histc or torch.bincount for efficient computation
            indices = self.num_classes * targets + preds
            conf_matrix = torch.bincount(
                indices,
                minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)
            self.matrix += conf_matrix

    def compute(self) -> torch.Tensor:
        """
        Returns the confusion matrix.

        Returns:
            torch.Tensor: The confusion matrix.
        """
        return self.matrix


# ------------------- Regression Metrics -------------------


class MeanSquaredError(Metric):
    """
    Computes the Mean Squared Error (MSE) regression loss.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        self.sum_squared_error: float = 0.0
        self.total: int = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.
        """
        with torch.no_grad():
            # Use F.mse_loss without reduction
            mse = F.mse_loss(outputs, targets, reduction='sum')
            self.sum_squared_error += mse.item()
            self.total += targets.numel()

    def compute(self) -> float:
        """
        Computes the Mean Squared Error.

        Returns:
            float: The computed MSE.
        """
        return self.sum_squared_error / self.total if self.total > 0 else 0.0


class MeanAbsoluteError(Metric):
    """
    Computes the Mean Absolute Error (MAE) regression loss.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        self.sum_absolute_error: float = 0.0
        self.total: int = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.
        """
        with torch.no_grad():
            # Use F.l1_loss without reduction
            mae = F.l1_loss(outputs, targets, reduction='sum')
            self.sum_absolute_error += mae.item()
            self.total += targets.numel()

    def compute(self) -> float:
        """
        Computes the Mean Absolute Error.

        Returns:
            float: The computed MAE.
        """
        return self.sum_absolute_error / self.total if self.total > 0 else 0.0


class RootMeanSquaredError(Metric):
    """
    Computes the Root Mean Squared Error (RMSE) regression loss.
    """

    def __init__(self) -> None:
        self.mse_metric = MeanSquaredError()

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        self.mse_metric.reset()

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.
        """
        self.mse_metric.update(outputs, targets)

    def compute(self) -> float:
        """
        Computes the Root Mean Squared Error.

        Returns:
            float: The computed RMSE.
        """
        mse = self.mse_metric.compute()
        return mse ** 0.5


class R2Score(Metric):
    """
    Computes the R-squared (coefficient of determination) regression score.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        self.sum_squared_errors: float = 0.0
        self.sum_squared_total: float = 0.0
        self.targets_mean: float = 0.0
        self.total: int = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal state with new data.

        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.
        """
        with torch.no_grad():
            self.total += targets.numel()
            # Incremental mean calculation
            delta = targets - self.targets_mean
            self.targets_mean += delta.sum().item() / self.total

            residuals = targets - outputs
            self.sum_squared_errors += (residuals ** 2).sum().item()
            total_variance = (targets - self.targets_mean) ** 2
            self.sum_squared_total += total_variance.sum().item()

    def compute(self) -> float:
        """
        Computes the R-squared score.

        Returns:
            float: The computed R-squared score.
        """
        if self.sum_squared_total == 0.0:
            return 0.0
        return 1 - (self.sum_squared_errors / (self.sum_squared_total + 1e-7))
