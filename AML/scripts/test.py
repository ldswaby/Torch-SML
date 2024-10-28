import sys

sys.path.append('..')  # Adjust as necessary to import modules from parent directory

import yaml
from data.datasets import CustomDataset

from models.custom_model import CustomModel
from trainers.evaluator import Evaluator

# Load configuration
with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize components
dataset = CustomDataset(config['dataset']['params'])
model = CustomModel(config['model']['params'])

# Load trained model weights if saved
# model.load_state_dict(torch.load('path_to_model.pth'))

# Initialize evaluator and start evaluation
evaluator = Evaluator(model, dataset, config['training'])
evaluator.evaluate()