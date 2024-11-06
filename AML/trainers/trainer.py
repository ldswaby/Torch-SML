from AML.config import Config
from AML.loss import _build_loss
from AML.metrics import _build_metrics
from AML.models import _build_model


class Trainer:
    """Object for automatically handling training given config
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = self._build_model()
        self.loss = self._build_loss()
        self.metrics = self._build_metrics()

    def _build_loss(self):
        return _build_loss(self.config)

    def _build_metrics(self):
        return _build_metrics(self.config)

    def _build_model(self):
        return _build_model(self.config)
