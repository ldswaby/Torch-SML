from torch.nn import Module

from ..utils.registry import Registry

MODEL_REGISTRY = Registry('Model')

from .custom_model import MyCustomModel
from . import vision, wrappers


def _build_model(config: dict) -> Module:
    """Returns CompositeLoss object

    Args:
        config (dict): _description_

    Returns:
        dict: _description_
    """
    model = MODEL_REGISTRY.get(config['MODEL']['name'])(**config['MODEL']['kwargs'])
    return wrappers.ClassificationModel(model)


del Module, Registry