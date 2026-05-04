import torch
import gc
import logging
import wandb
import time
import os
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch_geometric.graphgym.config import cfg
from torch.optim.lr_scheduler import OneCycleLR
from train.metrics import compute_metrics_and_logging, compute_loss
from torch.optim import swa_utils


def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/"
    """
    prefixes = ['train', 'val']
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        result.update({f"{prefixes[i]}/{k}": v for k, v in stats.items()})
    return result

def train(model, loaders, optimizer, loggers):
    """
    Args:
        model: PyTorch model
        loaders: List of PyTorch data loaders
        optimizer: PyTorch optimizer
        loggers: List of loggers

    """

    # run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.name, config=cfg)  # mode="online"
    run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.name, config=cfg, mode="offline")

    num_splits = len(loggers)
    full_epoch_times = []
    perf = [[] for _ in range(num_splits-1)]  # Only record the log information of train and val
    ckpt_dir = osp.join(cfg.run_dir, "ckpt/")  # results/xxx/ckpt/

    if cfg.batch_accumulation > 1:
        scheduler = OneCycleLR(optimizer,
                               max_lr=cfg.lr,
                               total_steps=cfg.optim.max_epoch * len(loaders[0]) // cfg.batch_accumulation + cfg.optim.max_epoch,
                               pct_start=cfg.warmup)
    else:
        scheduler = OneCycleLR(optimizer,
                               max_lr=cfg.lr,
                               total_steps=cfg.optim.max_epoch * len(loaders[0]) // cfg.batch_accumulation,
                               pct_start=cfg.warmup)

    if cfg.useSWA:
        swa_model = swa_utils.AveragedModel(model)

    for cur_epoch in range(cfg.optim.max_epoch):
        start_time = time.perf_counter()
        
        train_epoch(loggers[0], loaders[0], model, optimizer, cfg.batch_accumulation, scheduler, cur_epoch=cur_epoch)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        eval_epoch(loggers[1], loaders[1], model)
        perf[1].append(loggers[1].write_epoch(cur_epoch))

        full_epoch_times.append(time.perf_counter() - start_time)
        run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.     
        best_epoch = int(np.array([vp['MAE'] for vp in perf[1]]).argmin())
        best_train = f"train_MAE: {perf[0][best_epoch]['MAE']:.4f}"
        best_val = f"val_MAE: {perf[1][best_epoch]['MAE']:.4f}"
        bstats = {"best/epoch": best_epoch}
        for i, s in enumerate(['train', 'val']): 
            bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
            bstats[f"best/{s}_MAE"] = perf[i][best_epoch]['MAE']
        logging.info(bstats)
        run.log(bstats, step=cur_epoch)

        run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
        run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)

        if cfg.useSWA and cur_epoch + cfg.swa_epochs >= cfg.optim.max_epoch:  # SWA
            swa_model.update_parameters(model)

        # Checkpoint the best epoch params.
        if best_epoch == cur_epoch:
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            
            os.makedirs(ckpt_dir, exist_ok=True)  # ckpt_dir = "results/jarvis_dft_3D_formation_energy_peratom/seed/ckpt/"
            ckpt_path = osp.join(ckpt_dir, 'best.ckpt')
            
            torch.save(ckpt, ckpt_path)
        
            logging.info(f"Best checkpoint saved at {ckpt_path}")

        logging.info(
            f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
            f"(avg {np.mean(full_epoch_times):.1f}s) | "
            f"Best so far: epoch {best_epoch}\t"
            f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
            f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        )

    # gc.collect()
    if cfg.useSWA:  # It is called at the end of training, specifically for updating the Batch Normalization (BN) layer for the SWA model.
        print("Updating BNs for Stochastic Weight Averaging")
        device = swa_model.parameters().__next__().device
        swa_utils.update_bn(loaders[0], swa_model, device=device)
        torch.save(swa_model.state_dict(), osp.join(ckpt_dir, 'swa.ckpt'))

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])

    eval_epoch(loggers[-1], loaders[-1], model, test_metrics=True)

    perf_test = loggers[-1].write_epoch(best_epoch)
    best_test = f"test_MAE: {perf_test['MAE']:.4f}"
    run.log({f"test/{k}": v for k, v in perf_test.items()})
    bstats[f"best/test_loss"] = perf_test['loss']
    bstats[f"best/test_MAE"] = perf_test['MAE']
    logging.info(bstats)
    run.log(bstats)

    logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf_test['loss']:.4f} {best_test}"
            )

    if cfg.useSWA:
        swa_ckpt = torch.load(osp.join(ckpt_dir, 'swa.ckpt'))
        swa_model.load_state_dict(swa_ckpt)
        aver_MAE = swa_eval_epoch(loaders[-1], swa_model)
        logging.info(f"SWA test MAE: {aver_MAE:.4f}")

    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")

    for logger in loggers:
        logger.close()
   
    logging.info('Task done, results saved in %s', ckpt_dir)

    run.finish()


def train_epoch(logger, loader, model, optimizer, batch_accumulation, scheduler, **kwargs):
    """
    Train the model for one epoch.
    Args:
        logger (Logger): Logger object to log training information.
        loader (DataLoader): DataLoader object providing the training data.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): Optimizer for updating the model parameters.
        batch_accumulation (int): Number of batches to accumulate gradients before updating the model parameters.
        scheduler (Scheduler): Learning rate scheduler.
    Raises:
        Exception: If the specified loss function is not implemented.
    Returns:
        None
    """
    model.train()
    optimizer.zero_grad()
    cur_epoch = kwargs.get('cur_epoch', 0)

    for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
        time_start = time.time()
        batch.to("cuda:0")

        pred, true = model(batch)
            
        MAE, MSE = compute_loss(pred, true)

        if cfg.loss == "MAE":
            loss = MAE
        elif cfg.loss == "MSE":
            loss = MSE
        else:
            raise Exception("Loss not implemented")

        loss.mean().backward()

        # Update the model parameters according to the gradient accumulation strategy and adjust the learning rate.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            optimizer.step()
            if not cfg.useSWA or cfg.useSWA and cur_epoch + cfg.swa_epochs < cfg.optim.max_epoch:
                scheduler.step()
            optimizer.zero_grad()

        compute_metrics_and_logging(pred=pred.detach(),
                                    true=true.detach(),
                                    mae=MAE.detach(),
                                    mse=MSE.detach(),
                                    loss=loss.detach(),
                                    lr=optimizer.param_groups[0]['lr'],
                                    time_used=time.time()-time_start,
                                    logger=logger)


def eval_epoch(logger, loader, model, test_metrics=False):
    """
    Evaluate the model for one epoch.
    Args:
        logger (Logger): Logger object for logging metrics and information.
        loader (DataLoader): DataLoader object providing the dataset.
        model (nn.Module): The model to be evaluated.
        test_metrics (bool, optional): Flag to indicate if test metrics should be computed. Defaults to False.
    Raises:
        Exception: If the specified loss function in the configuration is not implemented.
    Returns:
        None
    """
    model.eval()
    
    with torch.no_grad():

        for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
            time_start = time.time()
            batch.to("cuda:0")

            pred, true = model(batch)

            MAE, MSE = compute_loss(pred, true)

            if cfg.loss == "MAE":
                loss = MAE
            elif cfg.loss == "MSE":
                loss = MSE
            else:
                raise Exception("Loss not implemented")

            compute_metrics_and_logging(pred=pred.detach(),
                                        true=true.detach(),
                                        mae=MAE.detach(),
                                        mse=MSE.detach(),
                                        loss=loss.detach(),
                                        lr=0,
                                        time_used=time.time()-time_start,
                                        logger=logger,
                                        test_metrics=test_metrics)


def swa_eval_epoch(loader, model):
    model.eval()

    with torch.no_grad():
        sum_MAE = 0
        for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
            batch.to("cuda:0")

            pred, true = model(batch)

            MAE, MSE = compute_loss(pred, true)

            sum_MAE += MAE.detach().mean().item()

    return sum_MAE / len(loader)
