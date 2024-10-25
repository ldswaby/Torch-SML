from trainers.trainer import Trainer
from optimizers.custom_optimizer import CustomOptimizer
from models.custom_model import CustomModel
from losses.custom_loss import CustomLoss
from callbacks.custom_callback import CustomCallback
from data.datasets import CustomDataset
import yaml
import sys

# Adjust as necessary to import modules from parent directory
sys.path.append('..')


# Load configuration

with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize components
dataset = CustomDataset(config['dataset']['params'])
model = CustomModel(config['model']['params'])
loss_fn = CustomLoss(config['loss']['params'])
optimizer = CustomOptimizer(model.parameters(), config['optimizer']['params'])
callbacks = [CustomCallback(config['callbacks'][0]['params'])]

# Initialize trainer and start training
trainer = Trainer(model, dataset, loss_fn, optimizer,
                  callbacksz, config['training'])
trainer.train()
