import inspect
import torchvision.transforms as T
from torchvision.transforms import *

from .. import TRANSFORM_REGISTRY

for name, obj in inspect.getmembers(T):
    if inspect.isclass(obj):
        TRANSFORM_REGISTRY.register(name=name)(obj)
