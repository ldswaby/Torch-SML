import torch
from torch.utils.data import DataLoader

class Evaluator:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')