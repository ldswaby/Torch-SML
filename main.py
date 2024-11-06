"""
These should also inherit from Metric (as this inherits from nn.Module already)
"""
import yaml

from AML import datasets
from AML.config import Config
from AML.loss import _build_loss
from AML.metrics import _build_metrics
from AML.models import _build_model
from AML.trainers import Trainer

cfg = Config('AML/config/multiclass_classification_config.yaml')

model = _build_model(config=cfg)
metrics = _build_metrics(config=cfg)
criterion = _build_loss(config=cfg)
trainer = Trainer(cfg)

breakpoint()

train_metrics = metrics['Train']
val_metrics = metrics['Validation']
test_metrics = metrics['Test']
