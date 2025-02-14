from typing import Callable, List, Optional, Union
from contextlib import nullcontext

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from AML.callbacks import Callback, CallbackList
from AML.callbacks.utils import _process_callbacks
from AML.metrics import Metric, MetricCollection, _process_metrics
from AML.utils.training import TrainingProgressBar


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
):
    """Train a model for one epoch over the given DataLoader.

    Args:
        model (nn.Module): The model to train.
        trainloader (DataLoader): The DataLoader for training data.
        optimizer (Optimizer): The optimizer to update model parameters.
        criterion (Callable): A callable for computing loss.
        device (torch.device, optional): The device to run on. Defaults to CPU.
        lr_scheduler (Optional[_LRScheduler], optional): A learning rate scheduler. Defaults to None.
        metrics (Optional[Union[List[Metric], MetricCollection]], optional): Metrics to compute. Defaults to None.
        callbacks (Optional[Union[List[Callback], CallbackList]], optional): Callback objects. Defaults to None.
        pbar (Optional[TrainingProgressBar]): The progress bar object, if any.

    Returns:
        dict: Dictionary of computed metrics or logs (e.g. {'loss': ..., 'acc': ...}).
    """
    metrics = _process_metrics(metrics)
    callbacks = _process_callbacks(callbacks, model)

    model.to(device)
    model.train()
    criterion.set_device(device)
    metrics.reset()
    metrics.to(device)

    if pbar is not None:
        pbar.start_epoch(1)
        pbar.start_batch(total_batches=len(trainloader))

    for _, batch in enumerate(trainloader):
        # Move data onto device
        inputs = batch['data'].to(device)
        targets = batch['target'].to(device)

        callbacks.on_train_batch_begin(batch)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss['loss_total'].backward()
        optimizer.step()

        # Update metrics
        # TODO: write some logic to decide which outputs go into which metrics
        # function in which order (i.e. 'embeddings', 'logits', 'probs', 'preds').
        # Currently just assumes preds, like multiclass classification
        metrics.update(outputs['preds'], targets)

        # Optional: Step LR scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Gather logs to pass to the callback
        batch_logs = {
            'loss': loss['loss_total'].item(),
            # You can add more metrics here if needed
        }

        callbacks.on_train_batch_end(batch, batch_logs)

        # Update the batch progress bar (showing loss, for instance)
        if pbar is not None:
            pbar.update_batch(loss=loss['loss_total'].item())

    # End the batch-level progress bar
    if pbar is not None:
        pbar.end_batch()

    # Gather final epoch logs from metrics
    epoch_logs = metrics.compute()
    print(epoch_logs)

        # TODO: work on terminal output. Do we want eval metrics in there too?

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
    metrics: Optional[Union[List[Metric], MetricCollection]] = None,
    callbacks: Optional[Union[List[Callback], CallbackList]] = None,
    pbar: Optional[TrainingProgressBar] = None,
):
    """Main training loop that runs for a given number of epochs.

    Args:
        model (nn.Module): The model to train.
        trainloader (DataLoader): The DataLoader for training data.
        optimizer (Optimizer): The optimizer for updating parameters.
        criterion (Callable): A callable to compute the loss.
        epochs (int): Number of epochs to train.
        evalloader (Optional[DataLoader], optional): DataLoader for evaluation. Defaults to None.
        eval_interval (int, optional): Frequency of evaluation (in epochs). Defaults to 1.
        device (torch.device, optional): Device to run on. Defaults to CPU.
        lr_scheduler (Optional[_LRScheduler], optional): Learning rate scheduler. Defaults to None.
        metrics (Optional[Union[List[Metric], MetricCollection]], optional): Metrics to compute. Defaults to None.
        callbacks (Optional[Union[List[Callback], CallbackList]], optional): List of callbacks. Defaults to None.

    Returns:
        dict: Dictionary of final logs (e.g. {'loss': ..., 'acc': ...}).
    """
    metrics = _process_metrics(metrics)
    callbacks = _process_callbacks(callbacks, model)

    model.to(device)
    metrics.to(device)

    callbacks.on_train_begin(None)

    # Wrap everything in the TrainingProgressBar context:
    with pbar if pbar is not None else nullcontext():

        for epoch in range(1, epochs + 1):
            callbacks.on_epoch_begin(epoch)

            # Indicate new epoch to the progress bar
            pbar.start_epoch(current_epoch=epoch - 1)

            # Run one epoch of training
            train_logs = train_one_epoch(
                model=model,
                trainloader=trainloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                lr_scheduler=lr_scheduler,
                metrics=metrics,
                callbacks=callbacks,
                pbar=pbar,  # <-- pass the progress bar
            )

            # Advance the epoch progress bar
            pbar.update_epoch()
            pbar.end_epoch()

            # Evaluate if needed
            if evalloader and (epoch % eval_interval == 0):
                # Show evaluation progress in the bar
                pbar.start_eval(epoch_index=epoch - 1)
                # Do something to evaluate (this would be your evaluate function)
                # For example:
                # eval_logs = evaluate(model, evalloader, device, metrics)
                # Dummy simulate an accuracy:
                accuracy = 85.0
                pbar.end_eval(accuracy)

            callbacks.on_epoch_end(epoch, train_logs)

            # If a callback sets stop_training = True, break early
            if callbacks.stop_training:
                break

    callbacks.on_train_end(train_logs)
    return train_logs

# def train(
#     model,
#     trainloader: DataLoader,
#     optimizer: Optimizer,
#     criterion: Callable,
#     epochs: int,
#     evalloader: Optional[DataLoader] = None,
#     eval_interval: int = 1,
#     device: torch.device = torch.device('cpu'),
#     lr_scheduler: Optional[_LRScheduler] = None,
#     metrics: Optional[Union[List[Metric], MetricCollection]] = None,
#     callbacks: Optional[Union[List[Callback], CallbackList]] = None
# ):
#     """_summary_

#     Args:
#         model (_type_): _description_
#         trainloader (DataLoader): _description_
#         optimizer (Optimizer): _description_
#         criterion (Callable): _description_
#         epochs (int): _description_
#         evalloader (Optional[DataLoader], optional): _description_. Defaults to None.
#         eval_interval (int, optional): _description_. Defaults to 1.
#         device (torch.device, optional): _description_. Defaults to torch.device('cpu').
#         lr_scheduler (Optional[_LRScheduler], optional): _description_. Defaults to None.
#         metrics (Optional[Union[List[Metric], MetricCollection]], optional): _description_. Defaults to None.
#         callbacks (Optional[Union[List[Callback], CallbackList]], optional): _description_. Defaults to None.

#     Returns:
#         _type_: _description_
#     """
#     # Process callables
#     metrics = _process_metrics(metrics)
#     callbacks = _process_callbacks(callbacks, model)

#     model.to(device)
#     metrics.to(device)

#     callbacks.on_train_begin(None)

#     for epoch in range(1, epochs + 1):

#         callbacks.on_epoch_begin(epoch)

#         # train
#         _logs = train_one_epoch(
#             model, trainloader, optimizer, criterion, device, lr_scheduler, metrics,
#             callbacks
#         )

#         if evalloader and epoch % eval_interval == 0:
#             # TODO
#             pass

#         callbacks.on_epoch_end(epoch, _logs)

#         if callbacks.stop_training:
#             break

#     callbacks.on_train_end(_logs)

#     return _logs
