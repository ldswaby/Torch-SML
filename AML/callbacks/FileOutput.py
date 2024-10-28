import warnings

from AML.callbacks import Callback


class FileOutput(Callback):
    """Abstract class for callbacks that generate files"""

    def __init__(self, out_dir: str, save_every: int = 1) -> None:
        if not isinstance(save_every, int):
            raise ValueError(
                f"Unrecognized save_every: {save_every}. Expected interger value."
            )
        if save_every <= 0:
            warnings.warn(
                f'Expected save_every to be > 0. Received {save_every}. Fallback to 1'
            )
            save_every = 1

        self.out_dir = out_dir
        self.save_every = save_every
        return
