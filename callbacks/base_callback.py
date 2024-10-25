import warnings
from typing import Optional

class Callback:
    """Base class used to build new callbacks.
    CalLbacks can be passed to AML train/test functions in order to hook into
    the various stages of the model training, evaluation, and inference
    Lifecycle.

    To create a custom calLback, subclass AML.base.Callback and override the
    method associated with the stage of interest.

    If you want to use "Callback objects in a custom training Loop:
    1. You should pack all your callbacks into a single "callbacks.
        CalLbackList* so they can all be called together.
    2. You will need to manually call all the "on_* methods at the appropriate
        Locations in your Loop.

    The "Logs dictionary that callback methods take as argument will contain
    keys for quantities relevant to the current batch or epoch (see
    method-specific docstrings).
    """

    def on_train_batch_begin(self, batch, logs: Optional[dict] = None):
        """CalLed at the beginning of a training batch in train methods/
        functions. Subclasses should override for any actions to run.
        Args:
            batch (dict): Batch within the current epoch.
            logs (Optional [dict], optional): Currently no data is passed to
                this argument for this method but that may change in the
                future. Defaults to None.
        """

    def on_train_batch_end(self, batch, logs: Optional[dict] = None):
        """Called at the beginning of a training batch in train methods
        functions. Subclasses should override for any actions to run.

        Args:
            batch (dict): Batch within the current epoch.
            logs (Optional [dict], optional): Results for train batch. Contains
                keys ['outputs', embeddings', 'Loss', 'preds', 'correct',
                'acc']. Defaults to None.
        """

    def on_test_batch_begin(self, batch, logs: Optional[dict] = None):
        """Called at the beginning of a testing batch in "test" methods/
        functions. Subclasses should override for any actions to run.

        Args:
            batch (dict): Test batch.
            Logs (Optional[dict], optional): Currently no data is passed to
                this argument for this method but that may change in the
                future. Defaults to None.
        """

    def on_test_batch_end (self, batch, logs: Optional[dict] = None):
        """Called at the end of a batch in "test" methods. Also called at the
        end of a validation batch in the train™ methods, if validation data is
        provided. Subclasses should override for any actions to run.

        Args:
            batch (dict): Test/validation batch
            logs (Optional [dict], optional): Results for current batch.
                Contains keys ['targets', 'outputs', 'embeddings', 'Losses",
                'preds']. Defaults to None.
        """

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        """Called at the start of an epoch. Subclasses should override for any
        actions to run. This function should only be cälled during TRAIN mode.

        Args:
            epoch (int): index of epoch.
            logs (Optional [dict], optional): Contains keys ['run_id']. Defaults
            to None.
        """

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

    def on_train_begin(self, logs: Optional[dict] = None):
        """Called at the beginning of training. Subclasses should override for
        any actions to run.

        Args:
            logs (Optional [dict], optional): Currently no data is passed to
                this argument for this method but that may change in the
                future. Defaults to None.
        """

    def on_train_end(self, logs: Optional[dict] = None):
        """Called at the end of training. Subclasses should override for any
        actions to run.

        Args:
            Logs (Optional [dict], optional): Contains keys ['run id', 'epoch',
                train _Loss', 'train_acc", "val_loss', 'val_acc']. Defaults to
                None.
        """

    def on_test_begin(self, logs: Optional[dict] = None):
        """Called at the beginning of evaluation or validation. Subclasses
        should override for any actions to run.

        Args:
            Logs (Optional[dict], optional): Currently no data is passed to
                this argument for this method but that may change in the
                future. Defaults to None.
        """

    def on_test_end(self, logs: Optional[dict] = None):
        """Called at the end of evaluation or validation. Subclasses should
        override for any actions to run.

        Args:
            logs (Optional[dict], optional): Contains keys ['targets', 'preds',
                'outputs', 'embeddings', 'Losses']. Defaults to None.
        """

    def set_model (self, model):
        """Sets callback model

        Args:
            model (_type_): _description  #TODO
        """
        self.model = model

    @property
    def contains_model(self) -> bool:
        """Checks if base class contains model

        Returns:
            bool: boolean flag indicating whether model is stored in callback
        """
        return hasattr(self, ' _model')

    @property
    def model(self):
        if not self.contains_model:
            warnings.warn(
                f'{self.__class__.__name__} instance has no _model attribute,'
                f' which may lead to unexpected behaviour. For proper use of'
                f' this property, set a model using the set _model method')
            return
        return self.model

    @property
    def stop_training(self) -> bool:
        """Flag to signal whether training should be stopped in training Loop

        Returns:
            bool
        """
        return self._stop_training if hasattr(self, '_stop_training') else False
