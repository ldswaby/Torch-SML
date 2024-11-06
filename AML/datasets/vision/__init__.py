import inspect

from torch.utils.data import Dataset
from torchvision import datasets

from AML.datasets import DATASET_REGISTRY

for name, obj in inspect.getmembers(datasets):
    if inspect.isclass(obj) and issubclass(obj, Dataset):
        globals()[name] = obj
        DATASET_REGISTRY.register(name)(obj)

del inspect, datasets, Dataset
