import multiprocessing as mp
from types import ModuleType
from typing import Dict, Optional, Type

import torch

__all__ = [
    'set_torch_device',
    'set_num_workers',
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
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_num_workers(device: torch.device) -> int:
    """Heuristically choose an appropriate num_workers for DataLoader.

    This function uses the total number of CPU cores and
    applies different heuristics depending on whether we're
    using a GPU (CUDA), Apple's Metal Performance Shaders (MPS),
    or just the CPU.

    Args:
        device (torch.device): The device where the model is placed.

    Returns:
        int: A suggested number of workers for PyTorch DataLoader.
    """
    cpu_count = mp.cpu_count()

    # Basic rule-of-thumb adjustments
    if device.type == 'cuda':
        # When using a GPU, we usually can leverage more CPU workers
        # because data loading can be overlapped with GPU compute.
        # e.g., use 0.75 * total cores (rounded) or (cpu_count - 2).
        num_workers = max(1, int(cpu_count * 0.75))
    elif device.type == 'mps':
        # For Apple Silicon (MPS), the GPU and CPU may share resources,
        # so we might not want to oversaturate.
        # Use half the cores or (cpu_count - 2), whichever is greater.
        num_workers = max(1, min(cpu_count - 2, cpu_count // 2))
    else:
        # CPU-only training generally benefits from parallel loading,
        # but not as aggressively as GPU-based training.
        # A common approach is half the cores or (cpu_count - 1).
        num_workers = max(1, cpu_count // 2)

    return num_workers


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
