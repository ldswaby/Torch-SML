import inspect
from torch import nn
from torch.nn import Module

from ..utils.registry import Registry
from . import loss  # Import your own custom loss functions, if any
from .CompositeLoss import CompositeLoss

LOSS_REGISTRY = Registry('Loss')


# List of modules containing loss functions
modules = [
    nn,
    loss  # Register any additional custom loss modules here
]

# Register torch loss functions
for module in modules:
    for name, obj in inspect.getmembers(module):
        # Check if the attribute is a class and a subclass of torch.nn.Module
        if inspect.isclass(obj) and issubclass(obj, Module) and 'Loss' in name:
            # Register the loss with its class name
            try:
                globals()[name] = obj
                LOSS_REGISTRY.register(name=name)(obj)
            except KeyError:
                # Loss function already registered, skip
                continue


def _build_loss(config: dict) -> CompositeLoss:
    """Returns CompositeLoss object

    Args:
        config (dict): _description_

    Returns:
        dict: _description_
    """
    loss_fns = []
    weights = []

    for loss_cfg in config['TRAINING']['Loss']:
        loss_fn = LOSS_REGISTRY.get(loss_cfg['name'])(**loss_cfg['kwargs'])
        w = loss_cfg['weight']
        loss_fns.append(loss_fn)
        weights.append(w)

    return CompositeLoss(loss_fns, weights)

# Cleanup namespace
del module, modules, obj, name, nn
