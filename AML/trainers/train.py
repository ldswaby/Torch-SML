
from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from AML.callbacks import Callback, CallbackList
from AML.callbacks.CallbackList import _process_callbacks


def train_one_epoch(
    model,
    trainloader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    device: torch.device = torch.device('cpu'),
    lr_scheduler: Optional[_LRScheduler] = None,
    callbacks: Optional[Union[List[Callback], CallbackList]] = None
):
    """Trains a torch.nn.Module model for one epoch
    Angs:
    model(nn. Module): _descniption_ trainloader(DataLoader): description optimizer(Optimizer): _description criterion(Callable):
    _description_
    epoch desciption(Optional[str], optional):
    _description_. Defaults to None.
    Returns:
    """
    model.to(device)
    model.train()

    callbacks = _process_callbacks(callbacks, model)

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
            bepoch.set_postfix({
                'Loss': float,
                'Acc': float
            })

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
    callbacks: Optional[Union[List[Callback], CallbackList]] = None
):
    model.to(device)

    callbacks = _process_callbacks(callbacks, model)

    callbacks.on_train_begin(run_logs)

    for epoch in range(1, epochs + 1):

        callbacks.on_epoch_begin(epoch)

        # train
        train_logs = train_one_epoch(
            model, trainloader, optimizer, criterion, device, lr_scheduler, callbacks
        )
        epoch_logs.update({'train_' + k: v for k, v in train_logs.items()})

        if evalloader and epoch % eval_interval == 0:
            # TODO
            pass

        callbacks.on_epoch_end(epoch, epoch_logs)

        if callbacks.stop_training:
            break

    callbacks.on_train_end(run_logs)

    return run_logs
