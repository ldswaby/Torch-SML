"""
These should also inherit from Metric (as this inherits from nn.Module already)
"""

import yaml

from AML.config import Config
from AML.metrics import _build_metrics

cfg = Config(
    '/Users/lukeswaby/Desktop/CODING/AML/AML/config/multiclass_classification_config.yaml')

# Load the YAML file
with open("AML/config/multiclass_classification_config.yaml", "r") as file:
    config = yaml.safe_load(file)

metrics = _build_metrics(config=config)

train_metrics = metrics['Train']
val_metrics = metrics['Validation']
test_metrics = metrics['Test']

breakpoint()
