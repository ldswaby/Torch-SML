from typing import List, Optional, Union

from torch import nn

from AML.callbacks import Callback, CallbackList


def _process_callbacks(
    callbacks: Optional[Union[List[Callback], CallbackList]] = None,
    model: Optional[nn.Module] = None
):
    """Ensures all callbacks are in CallbackList object

    Args:
        callbacks (Optional[Union[List[Callback], CallbackList]], optional): _description_. Defaults to None.
        model (Optional[nn.Module], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    callbacks = callbacks or []
    if not isinstance(callbacks, CallbackList):
        callbacks = CallbackList(callbacks, model)
    else:
        # Set model if absent in any callbacks
        if not callbacks.contains_model:
            callbacks.set_model(model)
    return callbacks
