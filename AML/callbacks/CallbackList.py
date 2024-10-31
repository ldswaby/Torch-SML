import warnings
from typing import List, Optional, Union

from torch import nn

from AML.callbacks import Callback


class CallbackList(Callback):
    """Container: Runs all the methods for each of a List of callbacks

    Args:
        Callback(_type_): _description_
    """

    def __init__(
        self,
        callbacks: Optional[List[Callback]] = None,
        model: Optional[nn.Module] = None
    ) -> None:
        self.callbacks = callbacks or []
        self.set_model(model)
        return

    def on_train_batch_begin(self, batch, logs: Optional[dict] = None):
        """Called at the end of each batch in the training data.

        Args:
            batch: The current batch of data. kwargs: Additional keyword arguments for
            custom behavior.
        """
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs: Optional[dict] = None):
        """Called at the beginning of a training batch in train methods
        functions. Subclasses should override for any actions to run.

        Args:
            batch (dict): Batch within the current epoch.
            logs (Optional [dict], optional): Results for train batch. Contains
                keys ['outputs', embeddings', 'Loss', 'preds', 'correct',
                'acc']. Defaults to None.
        """
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs)

    def on_test_batch_begin(self, batch, logs: Optional[dict] = None):
        """Called at the beginning of a testing batch in "test" methods/
        functions. Subclasses should override for any actions to run.

        Args:
            batch (dict): Test batch.
            Logs (Optional[dict], optional): Currently no data is passed to
                this argument for this method but that may change in the
                future. Defaults to None.
        """
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch, logs: Optional[dict] = None):
        """Called at the end of a batch in "test" methods. Also called at the
        end of a validation batch in the train™ methods, if validation data is
        provided. Subclasses should override for any actions to run.

        Args:
            batch (dict): Test/validation batch
            logs (Optional [dict], optional): Results for current batch.
                Contains keys ['targets', 'outputs', 'embeddings', 'Losses",
                'preds']. Defaults to None.
        """
        for callback in self.callbacks:
            callback.on_test_batch_end(batch, logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        """Called at the start of an epoch. Subclasses should override for any
        actions to run. This function should only be cälled during TRAIN mode.

        Args:
            epoch (int): index of epoch.
            logs (Optional [dict], optional): Contains keys ['run_id']. Defaults
            to None.
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Called at the end of an epoch. Subclasses should override for any
        actions to run. This function should only be called during TRAIN mode.

        Args:
            epoch (int): index of epoch.
            Logs (Optional[dict], optional): results for this training epoch,
                and for the validation epoch if validation is performed. In the
                former case, will contain keys ['run_id", 'train_loss',
                "train_acc']. In the latter case, ['run_id', 'train_Loss',
                "train_acc', "val_Loss", "val_acc', 'val preds",
                "val_targets"]. Train result keys are prefixed with train_* and
                validation result keys are prefixed with val_*. Defaults to
                None.
        """
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs: Optional[dict] = None):
        """Called at the beginning of training. Subclasses should override for
        any actions to run.

        Args:
            logs (Optional [dict], optional): Currently no data is passed to
                this argument for this method but that may change in the
                future. Defaults to None.
        """
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[dict] = None):
        """Called at the end of training. Subclasses should override for any
        actions to run.

        Args:
            Logs (Optional [dict], optional): Contains keys ['run id', 'epoch',
                train _Loss', 'train_acc", "val_loss', 'val_acc']. Defaults to
                None.
        """
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs: Optional[dict] = None):
        """Called at the beginning of evaluation or validation. Subclasses
        should override for any actions to run.

        Args:
            Logs (Optional[dict], optional): Currently no data is passed to
                this argument for this method but that may change in the
                future. Defaults to None.
        """
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs: Optional[dict] = None):
        """Called at the end of evaluation or validation. Subclasses should
        override for any actions to run.

        Args:
            logs (Optional[dict], optional): Contains keys ['targets', 'preds',
                'outputs', 'embeddings', 'Losses']. Defaults to None.
        """
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def set_model(self, model):
        """Sets model for all callbacks

        Args:
            model (_type_): _description  #TODO
        """
        for callback in self.callbacks:
            callback.set_model(model)
        self._model = model

    @property
    def contains_model(self) -> bool:
        """Checks if all callbacks contain model

        Returns:
            bool: boolean flag indicating whether model is stored in callback
        """
        return all(callback.contains_model for callback in self.callbacks)

    @property
    def model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if not self.contains_model:
            warnings.warn(
                'One or more callbacks have no _model attribute,'
                ' which may lead to unexpected behaviour. For proper use of'
                ' this property, set a model attribute'
            )
            return
        return self._model

    @property
    def stop_training(self) -> bool:
        """Flag to signal whether training should be stopped in training Loop

        Returns:
            bool
        """
        return any(callback.stop_training for callback in self.callbacks)
