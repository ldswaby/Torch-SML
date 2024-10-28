import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, dataset, loss_fn, optimizer, callbacks, config):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.config = config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

    def train(self):
        for epoch in range(1, self.config['num_epochs'] + 1):
            self.model.train()
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                for callback in self.callbacks:
                    callback.on_batch_begin(batch_idx)

                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                for callback in self.callbacks:
                    callback.on_batch_end(batch_idx)

            avg_loss = running_loss / len(self.data_loader)
            print(
                f"Epoch [{epoch}/{self.config['num_epochs']}], Loss: {avg_loss:.4f}")

            for callback in self.callbacks:
                callback.on_epoch_end(epoch)
