
from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from AML.callbacks import Callback, CallbackList


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

    # Callback init
    callbacks = callbacks or []
    if not isinstance(callbacks, CallbackList):
        callbacks = CallbackList(callbacks, model)
    else:
        # Set model if absent in any callbacks
        if not callbacks.contains_model:
            callbacks.set_model(model)

    # batch loop
    with tqdm(trainloader, unit=' batch', colour='green') as bepoch:
        for batch in bepoch:
            if epoch_desciption:
                bepoch.set_description(epoch_desciption)

            # data onto device
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)

            callbacks.on_train_batch_begin(batch)
            batch_logs = {}

            # forward pass
            batch_logs['outputs'], batch_logs['embeddings'] = model(inputs)
            # compute loss
            # TODO: This needs to be modified to reflect however
            # our loss library ends up working. E.g. will it also accept
            # embeddings?
            loss = criterion(batch_logs['outputs'], targets)
            batch_logs['loss'] = loss.mean()

            # NOTE: this is running loss. Update to method Alex suggested?
            epoch_loss.update(batch_logs['loss'].item(), targets.size(0))
            # backward pass and update weights
            optimizer.zero_grad()
            batch_logs['loss'].backward()
            optimizer.step()

            # Schedulen step
            if lr_scheduler:
                lr_scheduler.step()

            callbacks.on_train_batch_end(batch, batch_logs)

            # informative output
            bepoch.set_postfix({
                'Loss': float,
                'Acc': float
            })

    return {'loss': epoch_loss.value, 'acc': epoch_acc.value}
