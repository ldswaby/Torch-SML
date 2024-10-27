Repo for ML

# TODO:

* Callbacks:
  * `CallbackList`
  * `TensorboardWriter`
* `train.py`
  * Module/script containing `train_one_epoch` and `train` as well as a p`parse_args` etc
* `evaluate.py`
* Need a way of constructing the logs object passed through the callbacks.
  * Should be flexible and contain the outputs of any model. E.g. predictions for classification/regression/embeddings
  * There should be a `Metrics` object to track everything relevant to the problem, from accuracy and F1 scores to ROC curves and loss values. Configurable from yaml.
  * Model should output a dict containing subset of keys `['pred_classes', 'pred_values', 'logits', 'embeddings']`
  * Certain metrics will only be compatible with some of these outputs
