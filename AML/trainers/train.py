# train.py
from contextlib import nullcontext
from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from AML.callbacks import Callback, CallbackList
from AML.callbacks.utils import _process_callbacks
from AML.metrics import Metric, MetricCollection, _process_metrics
from AML.utils.training import TrainingProgressBar
from AML.trainers.test import test


def train_one_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    device: torch.device = torch.device('cpu'),
    lr_scheduler: Optional[_LRScheduler] = None,
    metrics: Optional[Union[List[Metric], MetricCollection]] = None,
    callbacks: Optional[Union[List[Callback], CallbackList]] = None,
    pbar: Optional[TrainingProgressBar] = None,
) -> dict:
    """Train a model for one epoch over the given DataLoader.

    Args:
        model (nn.Module): The model to train.
        trainloader (DataLoader): Dataloader for training.
        optimizer (Optimizer): Optimizer to update model params.
        criterion (Callable): Computes loss.
        device (torch.device, optional): Device to run on. Defaults to CPU.
        lr_scheduler (Optional[_LRScheduler], optional): LR scheduler. Defaults to None.
        metrics (Optional[Union[List[Metric], MetricCollection]], optional): Metrics. Defaults to None.
        callbacks (Optional[Union[List[Callback], CallbackList]], optional): Callbacks. Defaults to None.
        pbar (Optional[TrainingProgressBar], optional): Progress bar. Defaults to None.

    Returns:
        dict: Dictionary of logs (e.g. {'loss': float, ...}).
    """
    metrics = _process_metrics(metrics)
    callbacks = _process_callbacks(callbacks, model)

    # Set device
    for obj in [model, metrics]:
        obj.to(device)
    criterion.set_device(device)

    model.train()

    if pbar is not None:
        pbar.start_epoch(current_epoch=0)  # Or pass the actual epoch index if needed
        pbar.start_batch(total_batches=len(trainloader))

    for batch_index, batch in enumerate(trainloader):
        # Move data
        inputs = batch['data'].to(device)
        targets = batch['target'].to(device)

        callbacks.on_train_batch_begin(batch)

        # Forward / backward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss['loss_total'].backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update metrics
        metrics.update(outputs['preds'], targets)

        # Logging
        batch_logs = {'loss': loss['loss_total'].item()}
        callbacks.on_train_batch_end(batch, batch_logs)

        if pbar is not None:
            pbar.update_batch(loss=batch_logs['loss'])

    if pbar is not None:
        pbar.end_batch()

    epoch_logs = metrics.compute()
    print(epoch_logs)
    return epoch_logs


def train(
    model: nn.Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    epochs: int,
    evalloader: Optional[DataLoader] = None,
    eval_interval: int = 1,
    device: torch.device = torch.device('cpu'),
    lr_scheduler: Optional[_LRScheduler] = None,
    train_metrics: Optional[Union[List[Metric], MetricCollection]] = None,
    val_metrics: Optional[Union[List[Metric], MetricCollection]] = None,
    test_metrics: Optional[Union[List[Metric], MetricCollection]] = None,
    callbacks: Optional[Union[List[Callback], CallbackList]] = None,
    pbar: Optional[TrainingProgressBar] = None,
) -> dict:
    """Main training loop that runs for a given number of epochs.

    Wraps the entire training in a single context manager for pbar,
    then calls train_one_epoch to handle per-epoch training logic.

    Args:
        model (nn.Module): Model to train.
        trainloader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer.
        criterion (Callable): Loss function.
        epochs (int): Number of epochs.
        evalloader (Optional[DataLoader], optional): Eval data loader. Defaults to None.
        eval_interval (int, optional): Evaluate every N epochs. Defaults to 1.
        device (torch.device, optional): Device. Defaults to CPU.
        lr_scheduler (Optional[_LRScheduler], optional): LR scheduler. Defaults to None.
        metrics (Optional[Union[List[Metric], MetricCollection]], optional): Metrics. Defaults to None.
        callbacks (Optional[Union[List[Callback], CallbackList]], optional): Callbacks. Defaults to None.
        pbar (Optional[TrainingProgressBar], optional): Progress bar object. Defaults to None.

    Returns:
        dict: Final training logs.
    """
    # train_metrics = _process_metrics(train_metrics)
    # val_metrics = _process_metrics(val_metrics)
    # test_metrics = _process_metrics(test_metrics)
    callbacks = _process_callbacks(callbacks, model)

    # Move to device
    for obj in [model, train_metrics, val_metrics, test_metrics]:
        obj.to(device)

    callbacks.on_train_begin(None)

    # breakpoint()

    # If pbar is None, nullcontext() just does nothing
    with pbar if pbar is not None else nullcontext():
        for epoch in range(1, epochs + 1):
            callbacks.on_epoch_begin(epoch)

            # Start-of-epoch in the progress bar
            if pbar is not None:
                pbar.start_epoch(current_epoch=epoch)

            # Run one epoch
            train_logs = train_one_epoch(
                model=model,
                trainloader=trainloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                lr_scheduler=lr_scheduler,
                metrics=train_metrics,
                callbacks=callbacks,
                pbar=pbar,
            )

            # Mark epoch completion
            if pbar is not None:
                pbar.update_epoch(train_logs)
                pbar.end_epoch()

            # Evaluate if needed
            if evalloader and (epoch % eval_interval == 0):
                if pbar is not None:
                    pbar.start_eval(epoch)
                eval_logs = test(
                    model=model,
                    testloader=evalloader,
                    criterion=criterion,
                    device=device,
                    metrics=val_metrics,
                    callbacks=callbacks,
                    pbar=pbar
                )
                if pbar is not None:
                    pbar.end_eval(eval_logs)

            callbacks.on_epoch_end(epoch, train_logs)
            if callbacks.stop_training:
                break

    callbacks.on_train_end(train_logs)
    return train_logs