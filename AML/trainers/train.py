
from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from AML.callbacks import Callback, CallbackList
from AML.callbacks.utils import _process_callbacks
from AML.metrics import Metric, MetricCollection
from AML.metrics.utils import _process_metrics


def train_one_epoch(
    model,
    trainloader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    device: torch.device = torch.device('cpu'),
    lr_scheduler: Optional[_LRScheduler] = None,
    metrics: Optional[Union[List[Metric], MetricCollection]] = None,
    callbacks: Optional[Union[List[Callback], CallbackList]] = None
):
    """_summary_

    Args:
        model (_type_): _description_
        trainloader (DataLoader): _description_
        optimizer (Optimizer): _description_
        criterion (Callable): _description_
        device (torch.device, optional): _description_. Defaults to torch.device('cpu').
        lr_scheduler (Optional[_LRScheduler], optional): _description_. Defaults to None.
        metrics (Optional[Metric], optional): _description_. Defaults to None.
        callbacks (Optional[Union[List[Callback], CallbackList]], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Process callables
    metrics = _process_metrics(metrics)
    callbacks = _process_callbacks(callbacks, model)

    model.to(device)
    model.train()
    metrics.reset()
    metrics.to(device)

    # batch loop
    with tqdm(trainloader, unit=' batch', colour='green') as bepoch:

        for batch in bepoch:

            # if epoch_desciption:
            #     bepoch.set_description(epoch_desciption)

            # data onto device
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)

            callbacks.on_train_batch_begin(batch)

            # forward pass
            outputs = model(inputs)
            # compute loss
            # TODO: This needs to be modified to reflect however
            # our loss library ends up working. E.g. will it also accept
            # embeddings?
            loss = criterion(outputs, targets)

            # backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Scheduler step
            if lr_scheduler:
                lr_scheduler.step()

            callbacks.on_train_batch_end(batch, batch_logs)

            # informative output
            bepoch.set_postfix({'Loss': float, 'Acc': float})

    return {'loss': epoch_loss.value, 'acc': epoch_acc.value}


def train(
    model,
    trainloader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    epochs: int,
    evalloader: Optional[DataLoader] = None,
    eval_interval: int = 1,
    device: torch.device = torch.device('cpu'),
    lr_scheduler: Optional[_LRScheduler] = None,
    metrics: Optional[Union[List[Metric], MetricCollection]] = None,
    callbacks: Optional[Union[List[Callback], CallbackList]] = None
):
    """_summary_

    Args:
        model (_type_): _description_
        trainloader (DataLoader): _description_
        optimizer (Optimizer): _description_
        criterion (Callable): _description_
        epochs (int): _description_
        evalloader (Optional[DataLoader], optional): _description_. Defaults to None.
        eval_interval (int, optional): _description_. Defaults to 1.
        device (torch.device, optional): _description_. Defaults to torch.device('cpu').
        lr_scheduler (Optional[_LRScheduler], optional): _description_. Defaults to None.
        metrics (Optional[Union[List[Metric], MetricCollection]], optional): _description_. Defaults to None.
        callbacks (Optional[Union[List[Callback], CallbackList]], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Process callables
    metrics = _process_metrics(metrics)
    callbacks = _process_callbacks(callbacks, model)

    model.to(device)
    metrics.to(device)

    callbacks.on_train_begin(None)

    for epoch in range(1, epochs + 1):

        callbacks.on_epoch_begin(epoch)

        # train
        _logs = train_one_epoch(
            model, trainloader, optimizer, criterion, device, lr_scheduler, metrics,
            callbacks
        )

        if evalloader and epoch % eval_interval == 0:
            # TODO
            pass

        callbacks.on_epoch_end(epoch, _logs)

        if callbacks.stop_training:
            break

    callbacks.on_train_end(_logs)

    return _logs
