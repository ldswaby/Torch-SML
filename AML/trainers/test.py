from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from AML.callbacks import Callback, CallbackList
from AML.callbacks.CallbackList import _process_callbacks
from AML.metrics import Metric, MetricCollection
from AML.metrics.utils import _process_metrics


def test(
    model: nn.Module,
    testloader: DataLoader,
    criterion: Callable,
    device: torch.device = torch.device('cpu'),
    metrics: Optional[Union[List[Metric], MetricCollection]] = None,
    callbacks: Optional[Union[List[Callback], CallbackList]] = None
):
    # Process callables
    metrics = _process_metrics(metrics)
    callbacks = _process_callbacks(callbacks, model)

    model.to(device)
    model.eval()
    metrics.reset()

    callbacks.on_test_begin(None)

    with tqdm(testloader, desc='Eval', unit=' batch', colour='blue') as bepoch:

        with torch.no_grad():

            for batch in bepoch:

                callbacks.on_test_batch_begin(batch)

                # Data to device
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)

                # forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                metrics.update(outputs, targets)

                # TODO: print + build logs here

                callbacks.on_test_batch_end(batch, None)

            results = metrics.compute()

    callbacks.on_test_end(None)

    return results


def evaluate(model, dataloader, metrics, device='cpu'):
    """
    Evaluates the model on the given dataloader using the provided metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the evaluation data.
        metrics (torchmetrics.MetricCollection): A collection of metrics to compute.
        device (str or torch.device): The device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    model.eval()
    model.to(device)
    metrics.to(device)
    metrics.reset()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', colour='blue'):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # For classification tasks, get predictions
            if isinstance(outputs, torch.Tensor) and outputs.dim() > 1:
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = outputs
            # Update metrics
            metrics.update(preds, targets)
    # Compute metrics
    results = metrics.compute()
    return results
