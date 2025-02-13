from torchvision.transforms import Compose
from ..utils.registry import Registry

TRANSFORM_REGISTRY = Registry('Transform')

from . import vision


def _build_transforms(config: dict) -> Compose | None:
    transforms = []
    for _t in config['TRANSFORMS']:
        transforms.append(TRANSFORM_REGISTRY.get(_t['name'])(**_t['kwargs']))
    return Compose(transforms) if transforms else None
