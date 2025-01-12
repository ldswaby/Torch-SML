from AML.datasets import DATASET_REGISTRY
from AML.loss import LOSS_REGISTRY
from AML.metrics import METRIC_REGISTRY
from AML.models import MODEL_REGISTRY
from AML.callbacks import CALLBACK_REGISTRY
from AML.utils.data.splitters import DATA_SPLITTER_REGISTRY

library = {
    'datasets': DATASET_REGISTRY.list_keys(),
    'data_splitters': DATA_SPLITTER_REGISTRY.list_keys(),
    'losses': LOSS_REGISTRY.list_keys(),
    'models': MODEL_REGISTRY.list_keys(),
    'metrics': METRIC_REGISTRY.list_keys(),
    'callbacks': CALLBACK_REGISTRY.list_keys(),
}