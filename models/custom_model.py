from models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(BaseModel):
    def build_model(self):
        self.fc1 = nn.Linear(self.config['input_size'], 128)
        self.fc2 = nn.Linear(128, self.config['output_size'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x