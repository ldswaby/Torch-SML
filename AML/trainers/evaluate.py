import torch
from tqdm import tqdm


def evaluate(model, dataloader, metrics, device='cpu'):
    """
    Evaluates the model on the given dataloader using the provided metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the evaluation data.
        metrics (torchmetrics.MetricCollection): A collection of metrics to compute.
        device (str or torch.device): The device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    model.eval()
    model.to(device)
    metrics.to(device)
    metrics.reset()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', colour='blue'):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # For classification tasks, get predictions
            if isinstance(outputs, torch.Tensor) and outputs.dim() > 1:
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = outputs
            # Update metrics
            metrics.update(preds, targets)
    # Compute metrics
    results = metrics.compute()
    return results
