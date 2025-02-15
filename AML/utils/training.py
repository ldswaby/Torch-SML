# training_progress_bar.py
from typing import Dict, Any, Optional

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

__all__ = [
    'TrainingProgressBar'
]


class TrainingProgressBar:
    """Manages Rich progress bars for training, evaluation, and testing.

    An external training or testing loop calls these methods to display progress.

    Usage Example:
        >>> with TrainingProgressBar(total_epochs=10) as pbar:
        ...     # Train
        ...     for epoch in range(10):
        ...         pbar.start_epoch(epoch)
        ...         pbar.start_batch(total_batches=batches_per_epoch)
        ...         for batch in range(batches_per_epoch):
        ...             # training step ...
        ...             pbar.update_batch(loss=0.1234)
        ...         pbar.end_batch()
        ...         pbar.update_epoch()
        ...         pbar.end_epoch()
        ...
        ...     # Test
        ...     pbar.start_test(total_batches=test_batches)
        ...     for batch in range(test_batches):
        ...         # testing step ...
        ...         pbar.update_test(loss=0.5678)
        ...     pbar.end_test()
    """

    def __init__(self, total_epochs: int, eval_frequency: int = 1):
        """Initialize the training progress bar handler.

        Args:
            total_epochs (int): Total number of epochs to run.
            eval_frequency (int, optional): Frequency of evaluation in epochs.
                Defaults to 1 (evaluate every epoch).
        """
        self.total_epochs = total_epochs
        self.eval_frequency = eval_frequency

        self.progress: Progress = None
        self.epoch_task_id: int = None
        self.batch_task_id: int = None
        self.eval_task_id: int = None
        self.test_task_id: int = None

    def __enter__(self):
        """Enter context manager and create the Rich Progress object."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TextColumn("[yellow]{task.fields[extra]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        self.progress.__enter__()

        # Create the main epoch task.
        self.epoch_task_id = self.progress.add_task(
            "[cyan]Train",
            total=self.total_epochs,
            extra=""
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, close the Progress object."""
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    # -------------------------
    # PROCESSING METHODS
    # -------------------------
    def _logs2str(self, logs: Dict[str, Any]):
        out = ''
        for metric, value in logs.items():
            out += f"{metric.split('/')[-1]}: {value.item():.4f} "
        return out

    # -------------------------
    # EPOCH-LEVEL METHODS
    # -------------------------
    def start_epoch(self, current_epoch: int) -> None:
        """Prepare the display for a new epoch.

        Args:
            current_epoch (int): Index of the current epoch (0-based).
        """
        # Could reset or log a message here if desired.
        pass

    def update_epoch(self, epoch_logs: Dict[str, Any]) -> None:
        """Advance the epoch progress bar by one."""
        self.progress.update(
            self.epoch_task_id,
            advance=1,
            extra=self._logs2str(epoch_logs)
        )

    def end_epoch(self) -> None:
        """End the current epoch display (optional cleanup)."""
        # No removal needed, since the epoch task persists until all epochs complete.
        pass

    # -------------------------
    # BATCH-LEVEL METHODS (TRAIN)
    # -------------------------
    def start_batch(self, total_batches: int) -> None:
        """Create a new sub-task for training batches.

        Args:
            total_batches (int): Total number of batches in this epoch.
        """
        self.batch_task_id = self.progress.add_task(
            "[magenta]Epoch",
            total=total_batches,
            extra=""
        )

    def update_batch(self, loss: float) -> None:
        """Advance the train batch progress by one and display the loss.

        Args:
            loss (float): Current batch loss (or other metric).
        """
        if self.batch_task_id is not None:
            self.progress.update(
                self.batch_task_id,
                advance=1,
                extra=f"Loss: {loss:.4f}"
            )

    def end_batch(self) -> None:
        """Remove the train batch sub-task."""
        if self.batch_task_id is not None:
            self.progress.remove_task(self.batch_task_id)
            self.batch_task_id = None

    # -------------------------
    # EVALUATION METHODS
    # -------------------------
    # def start_eval(self, current_epoch: int) -> None:
    #     """Create a sub-task for evaluation.

    #     Args:
    #         current_epoch (int): Index of the current epoch (0-based).
    #     """
    #     self.eval_task_id = self.progress.add_task(
    #         f"[green]Eval (Epoch {current_epoch})",
    #         total=1,
    #         extra="Evaluating...",
    #         insert_before=self.epoch_task_id
    #     )

    # def end_eval(self, eval_logs: Dict[str, Any]) -> None:
    #     """Complete and remove the eval sub-task, then log results above the bars."""
    #     # if self.eval_task_id is not None:
    #         # self.progress.update(
    #         #     self.eval_task_id,
    #         #     advance=1,
    #         #     extra=self._logs2str(eval_logs)
    #         # )
    #         # self.progress.remove_task(self.eval_task_id)
    #         # self.eval_task_id = None

    #     # This logs a line *above* the live display (i.e., above the train bar)
    #     self.progress.log(
    #         f"[green bold]Eval results:[/green bold] {self._logs2str(eval_logs)}"
    #     )

    # -------------------------
    # TEST METHODS
    # -------------------------
    def start_test(self, total_batches: int, validation: bool=True) -> None:
        """Create a new sub-task for test batches (in a different color).

        Args:
            total_batches (int): Total number of test batches.
        """
        self.test_task_id = self.progress.add_task(
            "[blue]Eval" if validation else "[blue]Test",
            total=total_batches,
            extra=""
        )

        # self.test_task_id = self.progress.add_task(
        #     f"[red]Test",
        #     total=total_batches
        #     extra="Testing...",
        # )

    def update_test(self, loss: float) -> None:
        """Advance the test batch progress and display the loss.

        Args:
            loss (float): Current batch loss (or other metric).
        """
        if self.test_task_id is not None:
            self.progress.update(
                self.test_task_id,
                advance=1,
                extra=f"Loss: {loss:.4f}"
            )

    def end_test(
        self,
        test_logs: Dict[str, Any],
        validation: bool=True
    ) -> None:
        """Remove the test sub-task."""
        if self.test_task_id is not None:
            self.progress.remove_task(self.test_task_id)
            self.test_task_id = None

        if validation:
            self.progress.log(
                f"[green bold]Eval results:[/green bold] {self._logs2str(test_logs)}"
            )
        else:
            self.progress.log(
                f"[red bold]Test results:[/red bold] {self._logs2str(test_logs)}"
            )