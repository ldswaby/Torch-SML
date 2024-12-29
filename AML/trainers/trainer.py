from AML.config import Config
from AML.datasets import _build_dataset
from AML.loss import _build_loss
from AML.metrics import _build_metrics
from AML.models import _build_model
from AML.callbacks import _build_callbacks
from AML.utils.data.splitters import _data_splitter_factory
# from AML.trainers import train

# optimizer
# callbacks


class Trainer:
    """Object for automatically handling training given config
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset = self._build_dataset()
        self.model = self._build_model()
        self.loss = self._build_loss()
        self.metrics = self._build_metrics()
        self.callbacks = self._build_callbacks()

    def _build_dataset(self):
        """Builds and splits dataset according to config

        Returns:
            _type_: _description_
        """
        return _build_dataset(self.config)

    def _build_model(self):
        return _build_model(self.config)

    def _build_loss(self):
        return _build_loss(self.config)

    def _build_metrics(self):
        return _build_metrics(self.config)

    def _build_callbacks(self):
        return _build_callbacks(self.config)

    def __repr__(self):
        _v = vars(self)
        del _v['config']
        return str(_v)

    def train(self):
        raise NotImplementedError
