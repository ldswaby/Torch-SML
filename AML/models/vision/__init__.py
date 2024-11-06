from torchvision.models import list_models
from .. import MODEL_REGISTRY
from torchvision.models import *


# Register all torchvision models
for model_name in list_models():
    model_fn = get_model_builder(model_name)
    MODEL_REGISTRY.register(model_name)(model_fn)

del model_name, model_fn
