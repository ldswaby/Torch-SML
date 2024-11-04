from ..utils.registry import Registry
MODEL_REGISTRY = Registry('Model')

from .custom_model import MyCustomModel
from . import vision
