# test.py
from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from AML.callbacks import Callback, CallbackList
from AML.callbacks.utils import _process_callbacks
from AML.metrics import Metric, MetricCollection, _process_metrics
from AML.utils.training import TrainingProgressBar


def test(
    model: nn.Module,
    testloader: DataLoader,
    criterion: Callable,
    device: torch.device = torch.device('cpu'),
    metrics: Optional[Union[List[Metric], MetricCollection]] = None,
    callbacks: Optional[Union[List[Callback], CallbackList]] = None,
    pbar: Optional[TrainingProgressBar] = None,
    validation: bool=False
) -> dict:
    """Evaluates the model on a test dataset, using a shared TrainingProgressBar.

    If pbar is None, no Rich progress bar is shown.

    Args:
        model (nn.Module): Model to evaluate.
        testloader (DataLoader): Dataloader for test data.
        criterion (Callable): Loss function.
        device (torch.device): CPU or GPU.
        metrics (Optional[Union[List[Metric], MetricCollection]], optional): Metrics. Defaults to None.
        callbacks (Optional[Union[List[Callback], CallbackList]], optional): Callbacks. Defaults to None.
        pbar (Optional[TrainingProgressBar], optional): Rich progress bar. Defaults to None.
        validation: validation epoch or test epoch

    Returns:
        dict: Dictionary of computed metrics/results.
    """
    metrics = _process_metrics(metrics)
    callbacks = _process_callbacks(callbacks, model)

    model.to(device)
    model.eval()
    metrics.reset()
    metrics.to(device)

    callbacks.on_test_begin(None)

    # Don't open a second context manager: just call the pbar methods if available.
    if pbar is not None:
        pbar.start_test(total_batches=len(testloader), validation=validation)

    test_logs = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            callbacks.on_test_batch_begin(batch)

            # Move data
            inputs = batch['data'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update metrics, logs, etc.
            metrics.update(outputs['preds'], targets)
            test_logs = {'loss': loss['loss_total'].item()}

            callbacks.on_test_batch_end(batch, test_logs)

            if pbar is not None:
                pbar.update_test(loss=test_logs['loss'])


    test_logs = metrics.compute()
    if pbar is not None:
        pbar.end_test(test_logs, validation=validation)
    callbacks.on_test_end(test_logs)
    return test_logs