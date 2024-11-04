"""
These should also inherit from Metric (as this inherits from nn.Module already)
"""

import yaml

from AML.config import Config
from AML.metrics import _build_metrics

cfg = Config('AML/config/multiclass_classification_config.yaml')

metrics = _build_metrics(config=cfg)

train_metrics = metrics['Train']
val_metrics = metrics['Validation']
test_metrics = metrics['Test']
