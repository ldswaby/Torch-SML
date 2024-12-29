import os
from typing import Callable, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image

from AML.datasets import DATASET_REGISTRY, BaseDataset


@DATASET_REGISTRY.register('ImageDataset')
class ImageDataset(BaseDataset):
    """Dataset for loading images and targets from directories.

    Args:
        root (str): Root directory containing image folders.
        transform (Optional[Callable], optional): Transform to be applied on an image.
            Defaults to None.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None
    ) -> None:
        self.root = root
        self.transform = transform
        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }
        self.samples: List[Tuple[str, int]] = []

        for cls_name in self.classes:
            cls_folder = os.path.join(root, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    path = os.path.join(cls_folder, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def get_data(self, idx: int) -> torch.Tensor:
        path, _ = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Convert to tensor if no transform is provided
            image = torch.from_numpy(
                np.array(image)
            ).permute(2, 0, 1).float() / 255.0
        return image

    def get_target(self, idx: int) -> torch.Tensor:
        _, target = self.samples[idx]
        return torch.tensor(target)
