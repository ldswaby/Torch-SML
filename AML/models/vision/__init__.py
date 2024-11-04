from .. import MODEL_REGISTRY
from torchvision.models import *


# Register all torchvision models
for model_name in list_models():
    model_class = get_model_builder(model_name)
    MODEL_REGISTRY.register(model_name)(model_class)

del model_name, model_class