from torch import nn
from torch.nn import Module

from ..utils.registry import Registry
LOSS_REGISTRY = Registry('Loss')

from .CompositeLoss import CompositeLoss
from . import loss  # Import your own custom loss functions, if any

# List of modules containing loss functions
modules = [
    nn,
    loss  # Register any additional custom loss modules here
]

# Register torch loss functions
for module in modules:
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        # Check if the attribute is a class and a subclass of torch.nn.Module
        if isinstance(attr, type) and issubclass(attr, Module) and 'Loss' in attr_name:
            # Register the loss with its class name
            try:
                globals()[attr_name] = attr
                LOSS_REGISTRY.register(name=attr_name)(attr)
            except KeyError:
                # Loss function already registered, skip
                continue

# Cleanup namespace
del module, modules, attr, attr_name, nn
