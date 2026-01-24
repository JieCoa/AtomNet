import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg

# Metrics Config
l1_loss = nn.L1Loss(reduction="mean")
mse_loss = nn.MSELoss(reduction="mean")

def compute_loss(pred, true):
    """
    Computes the Mean Absolute Error (MAE) and Mean Squared Error (MSE) between the predicted and true values.

    Args:
        pred (Tensor): The predicted values.
        true (Tensor): The ground truth values.

    Returns:
        tuple: A tuple containing the MAE and MSE.
    """
    MAE = l1_loss(pred, true)
    MSE = mse_loss(pred, true)
    return MAE, MSE


@torch.no_grad()
def compute_metrics_and_logging(pred, true, mae, mse, loss, lr, time_used, logger, test_metrics=False):
    """
    Compute metrics and log the results using the provided logger.
    Parameters:
    pred (torch.Tensor): Predicted values.
    true (torch.Tensor): Ground truth values.
    mae (torch.Tensor): Mean Absolute Error.
    mse (torch.Tensor): Mean Squared Error.
    loss (torch.Tensor): Loss value.
    lr (float): Learning rate.
    time_used (float): Time used for the computation.
    logger (Logger): Logger object to update the stats.
    test_metrics (bool, optional): Flag to indicate if test metrics should be computed. Defaults to False.
    Returns:
    None
    """

    logger.update_stats(true=true.to("cpu"),
                        pred=pred.to("cpu"),
                        loss=loss.mean().item(),
                        MAE=mae.mean().item(),
                        MSE=mse.mean().item(),
                        lr=lr,
                        time_used=time_used,
                        params=cfg.params_count,
                        dataset_name=cfg.dataset.name
                        )