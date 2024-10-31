from types import ModuleType
from typing import Dict, Optional, Type

import torch

__all__ = [
    'set_torch_device',
    'fetch_pkg_subclasses'
]


def set_torch_device(device: Optional[str] = None) -> torch.device:
    """Sets the device for a training loop based on available hardware.

    Returns:
        torch.device: The device to be used for training, prioritized as:
                      1. CUDA GPU if available
                      2. Apple Metal (mps) for M1/M2 Macs if available
                      3. CPU as a fallback
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        print("Using CUDA GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple Metal (mps) on Mac.")
        return torch.device("mps")
    else:
        print("Using CPU.")
        return torch.device("cpu")


def fetch_pkg_subclasses(pkg: ModuleType, base_class: Type) -> Dict[str, Type]:
    """Lists the names and objects of subclasses of a specified base class within a package.

    Args:
        pkg (ModuleType): The package module to search within.
        base_class (Type): The base class to match subclasses against.

    Returns:
        Dict[str, Type]: A dictionary where the keys are the names of classes in the package
        that are subclasses of `base_class`, and the values are the class objects themselves.
    """
    return {
        name: obj for name, obj in pkg.__dict__.items()
        if isinstance(obj, type) and issubclass(obj, base_class)
    }
