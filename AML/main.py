"""
These should also inherit from Metric (as this inherits from nn.Module already)
"""

import yaml

from AML.metrics import build_metrics

# Load the YAML file
with open("AML/config/multiclass_classification_config.yaml", "r") as file:
    config = yaml.safe_load(file)

metrics = build_metrics(config=config)
