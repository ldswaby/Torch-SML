from typing import Dict, List, Optional

import yaml
from cerberus import Validator
from torch import optim

from AML.datasets import DATASET_REGISTRY
from AML.loss import LOSS_REGISTRY
from AML.metrics import METRIC_REGISTRY
from AML.models import MODEL_REGISTRY
from AML.callbacks import CALLBACK_REGISTRY
from AML.utils import fetch_pkg_subclasses
from AML.utils.data.splitters import DATA_SPLITTER_REGISTRY

# Optional vars
dataset_opts = DATASET_REGISTRY.list_keys()
model_opts = MODEL_REGISTRY.list_keys()
loss_opts = LOSS_REGISTRY.list_keys()
metric_opts = METRIC_REGISTRY.list_keys()
callback_opts = CALLBACK_REGISTRY.list_keys()
optimizer_opts = list(fetch_pkg_subclasses(optim, optim.Optimizer).keys())
data_splitter_opts = DATA_SPLITTER_REGISTRY.list_keys()
device_opts = ['cpu', 'cuda', 'mps']

# Arg checks
BOOL = {'type': 'boolean'}
STR = {'type': 'string'}
INT = {'type': 'integer'}
FLOAT = {'type': 'float'}
LIST = {'type': 'list'}
ANY = {'type': 'any'}

KWARGS_DICT = {
    'type': 'dict',
    'schema': {},
    'keysrules': STR,
    'allow_unknown': True,
    'default': {}
}


NATURAL_NUMBER = INT | {'min': 1}
WHOLE_NUMBER = INT | {'min': 0}
POSITIVE_REAL = FLOAT | {'min': 0.0}
REQUIRED = {'required': True}


def RANGE(min, max): return {'min': min, 'max': max}
def OPTIONS(opts): return {'allowed': opts}
def DEFAULT(value): return {'default': value}


def LIST_WITH_KWARGS(opts, **kwargs):
    return LIST | {
        'schema': {
            'type': 'dict',
            'schema': {
                'name': STR | OPTIONS(opts),
                **kwargs,
                'kwargs': KWARGS_DICT,

            }
        },
    }


SCHEMA = {
    'DATASET': {
        'type': 'dict',
        'schema': {
            'name': STR | OPTIONS(dataset_opts),
            'kwargs': KWARGS_DICT,
            'split_method': {
                'type': 'dict',
                'schema': {
                    'name': STR | OPTIONS(data_splitter_opts),
                    'kwargs': KWARGS_DICT
                }
            }
        },
    },
    'MODEL': {
        'type': 'dict',
        'schema': {
            'name': STR | OPTIONS(model_opts),
            'kwargs': KWARGS_DICT
        },
    },
    'TRAINING': {
        'type': 'dict',
        'schema': {
            'batch_size': INT,
            'Loss': LIST_WITH_KWARGS(loss_opts, weight=FLOAT),
            'Optimizer': {
                'type': 'dict',
                'schema': {
                    'name': STR | OPTIONS(optimizer_opts),
                    'kwargs': KWARGS_DICT
                }
            }
        },
    },
    'METRICS': {
        'type': 'dict',
        'schema': {
            'Train': LIST_WITH_KWARGS(metric_opts),
            'Test': LIST_WITH_KWARGS(metric_opts),
            'Validation': LIST_WITH_KWARGS(metric_opts),
        }
    },
    'CALLBACKS': LIST_WITH_KWARGS(callback_opts)
}


class Config(dict):

    def __init__(self, file_path: str):
        super().__init__()
        with open(file_path, 'r', encoding='utf-8') as f:
            self.update(yaml.safe_load(f))
        self._validate()
        return

    def _error_msg(self, msg: str, path: str) -> str:
        return f"Error in config field {path}: {msg}"

    def _format_errors(self, errs: Dict[str, List], path: Optional[List[str]] = None):
        msgs = []
        path = path or []

        for field, error_list in errs.items():
            curr_path = path + [field]
            for e in error_list:
                if isinstance(e, dict):
                    msgs.extend(self._format_errors(e, curr_path))
                elif isinstance(e, str):
                    msgs.append(self._error_msg(e, curr_path))
        return msgs

    def _custom_checks(self):
        """Additional custom arg checks should go here
        """
        msgs = []
        # if self['TRAINING']['epochs'] < self['VALIDATION']['interval']:
        #     msgs.append(self._error_msg(
        #         "Must be less than or equal to ['training epochs]'",
        #         ['validation', 'interval']
        #     ))
        return msgs

    def _validate(self):
        """Validate parsed config according to schema
        """
        errs = []
        v = Validator(SCHEMA)

        if not v.validate(self):
            errs.extend(self._format_errors(v.errors))

        # Custom checks
        errs.extend(self._custom_checks())

        if errs:
            raise ValueError('\n'.join(errs))

        # Add defaults
        self.update(v.normalized(self))
        return
