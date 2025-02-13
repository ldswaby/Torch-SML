from torch import optim

from AML.utils import set_torch_device
from AML.callbacks import _build_callbacks
from AML.config import Config
from AML.datasets import _build_dataset, _build_dataloaders
from AML.transforms import _build_transforms
from AML.loss import _build_loss
from AML.metrics import _build_metrics
from AML.models import _build_model
from AML.trainers.train import train, train_one_epoch


class Trainer:
    """Object for automatically handling training given config
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = set_torch_device()
        # self.transforms = self._build_transforms()
        self.datasets = self._build_dataset()
        self.dataloaders = self._build_dataloaders()
        self.model = self._build_model()
        self.loss = self._build_loss()
        self.optimizer = self._build_optimizer()
        self.metrics = self._build_metrics()
        self.callbacks = self._build_callbacks()

    # def _build_transforms(self):
    #     """Build data transforms
    #     """
    #     return _build_transforms(self.config)

    def _build_dataset(self):
        """Builds and splits dataset according to config

        Returns:
            _type_: _description_
        """
        return _build_dataset(self.config)

    def _build_dataloaders(self):
        """Builds and splits dataset according to config

        Returns:
            _type_: _description_
        """
        return _build_dataloaders(self.config, self.datasets, self.device)

    def _build_model(self):
        return _build_model(self.config)

    def _build_loss(self):
        return _build_loss(self.config)

    def _build_optimizer(self):
        return getattr(optim, self.config['TRAINING']['Optimizer']['name'])(
            self.model.parameters(), **self.config['TRAINING']['Optimizer']['kwargs']
        )

    def _build_metrics(self):
        return _build_metrics(self.config)

    def _build_callbacks(self):
        return _build_callbacks(self.config)

    def __repr__(self):
        _v = vars(self)
        del _v['config']
        return str(_v)

    def train_one_epoch(self):
        return train_one_epoch(
            model=self.model,
            trainloader=self.dataloaders['train'],
            optimizer=self.optimizer,
            criterion=self.loss,
            device=self.device,
            lr_scheduler=None,  # TODO
            metrics=self.metrics['train'],
            callbacks=self.callbacks,
            pbar=None,
        )

    def train(self):
        raise NotImplementedError
