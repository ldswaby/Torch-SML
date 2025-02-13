from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

__all__ = [
    'TrainingProgressBar'
]


class TrainingProgressBar:
    """Manages Rich progress bars for training (epoch/batch) and evaluation.

    This class separates the progress bar logic from the training loop.
    An external training loop calls these methods to display progress.

    Usage example:
        with TrainingProgressBar(total_epochs=10) as pbar:
            for epoch in range(total_epochs):
                pbar.start_epoch(epoch)
                pbar.start_batch(total_batches=batches_per_epoch)
                for _ in range(batches_per_epoch):
                    # ... training step
                    pbar.update_batch(loss=0.1234)
                pbar.end_batch()
                pbar.update_epoch()
                pbar.end_epoch()

                if (epoch + 1) % eval_frequency == 0:
                    pbar.start_eval(epoch)
                    # ... evaluation step
                    pbar.end_eval(accuracy=75.5)
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

        self.progress = None
        self.epoch_task_id = None
        self.batch_task_id = None
        self.eval_task_id = None

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
            expand=True
        )
        self.progress.__enter__()

        # Create the main epoch task.
        self.epoch_task_id = self.progress.add_task(
            "[cyan]Epochs",
            total=self.total_epochs,
            extra=""
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, close the Progress object."""
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def start_epoch(self, current_epoch: int) -> None:
        """Prepare the display for a new epoch.

        Args:
            current_epoch (int): Index of the current epoch (0-based).
        """
        # Optionally, you can show some message or reset states here.
        # The epoch progress bar itself is created in __enter__ (self.epoch_task_id).
        pass

    def update_epoch(self) -> None:
        """Advance the epoch progress bar by one."""
        self.progress.update(
            self.epoch_task_id,
            advance=1,
            extra="Epoch Complete"
        )

    def end_epoch(self) -> None:
        """End the current epoch display (optional cleanup)."""
        # Nothing to remove here, since the epoch task persists
        # until all epochs are complete.
        pass

    def start_batch(self, total_batches: int) -> None:
        """Create a new sub-task for batch iteration.

        Args:
            total_batches (int): Total number of batches in this epoch.
        """
        # Create a new task for batch-level progress
        self.batch_task_id = self.progress.add_task(
            "[magenta]Batches",
            total=total_batches,
            extra=""
        )

    def update_batch(self, loss: float) -> None:
        """Advance the batch progress by one and display the loss.

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
        """Remove the batch sub-task (cleanup after finishing an epoch)."""
        if self.batch_task_id is not None:
            self.progress.remove_task(self.batch_task_id)
            self.batch_task_id = None

    def start_eval(self, current_epoch: int) -> None:
        """Create a sub-task for evaluation.

        Args:
            current_epoch (int): Index of the current epoch (0-based).
        """
        self.eval_task_id = self.progress.add_task(
            f"[green]Evaluation (Epoch {current_epoch + 1})",
            total=1,
            extra="Evaluating..."
        )

    def end_eval(self, accuracy: float) -> None:
        """Complete the evaluation sub-task and remove it.

        Args:
            accuracy (float): Evaluation accuracy (or other metric).
        """
        if self.eval_task_id is not None:
            self.progress.update(
                self.eval_task_id,
                advance=1,
                extra=f"Accuracy: {accuracy:.2f}%"
            )
            self.progress.remove_task(self.eval_task_id)
            self.eval_task_id = None
