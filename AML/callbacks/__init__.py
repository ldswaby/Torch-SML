from ..utils.registry import Registry
CALLBACK_REGISTRY = Registry('Callback')

from .base_callback import Callback
from .CallbackList import CallbackList
from .Monitor import Monitor
from .FileOutput import FileOutput

from . import utils

def _build_callbacks(config: dict) -> CallbackList:
    callbacks = []
    for _c in config['CALLBACKS']:
        callbacks.append(CALLBACK_REGISTRY.get(_c['name'])(**_c['kwargs']))
    return CallbackList(callbacks)

del Registry
