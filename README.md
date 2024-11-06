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


# Models
  * Ouputs have to be of the format taken by classification metrics (e.g. for classification model using multiclass accuracy, outputs should be e.g. [2,1,2,1,0])
  * Models forward should output a dict, with keys:
    * `'outputs'`: Whatever format required by loss/metrics
    * `'logits'`: Unprocessed final layer outputs


# Loss
* preds can be logits for both loss and metrics
* In the train loop, loss fn inputs can be iteratively built up tp allow loss fns that take 2 or 3 inputs, as https://github.com/adambielski/siamese-triplet/blob/master/trainer.py#L65





## Problem

* Data outputs need to be in format required by training loop/model

Data (batches) => Model (logits/embeddings) => Loss/Metrics
