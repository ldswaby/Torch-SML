from ...registry import Registry
DATA_SPLITTER_REGISTRY = Registry('DataSplitter')

from .base_splitter import DataSplitter
from .holdout_splitter import HoldoutSplitter
from .stratified_splitter import StratifiedSplitter
from .kfold_splitter import KFoldSplitter
from .stratified_kfold_splitter import StratifiedKFoldSplitter

def _data_splitter_factory(config: dict) -> DataSplitter:
    """Returns CompositeLoss object

    Args:
        config (dict): _description_

    Returns:
        dict: _description_
    """
    splitter = DATA_SPLITTER_REGISTRY.get(
        config['DATASET']['split_method']['name'])(
            **config['DATASET']['split_method']['kwargs']
    )
    return splitter


# def splitter_factory(method: str, **kwargs) -> DataSplitter:
#     """
#     Factory function to create various splitter classes based on the specified method.

#     Args:
#         method (str): The splitting method.
#             Supported: ['holdout', 'stratified_holdout', 'kfold', 'stratified_kfold', 'time_based']
#         **kwargs: Additional keyword arguments for the splitter class.

#     Returns:
#         BaseSplitter: An instance of a concrete splitter class.
#     """
#     method = method.lower()
#     if method == "holdout":
#         return HoldoutSplitter(**kwargs)
#     elif method == "stratified":
#         return StratifiedSplitter(**kwargs)
#     elif method == "kfold":
#         return KFoldSplitter(**kwargs)
#     elif method == "stratified_kfold":
#         return StratifiedKFoldSplitter(**kwargs)
#     else:
#         raise ValueError(f"Unknown splitting method '{method}'.")