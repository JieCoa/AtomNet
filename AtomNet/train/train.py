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
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

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
    Train the model

    Args:
        model: PyTorch model
        loaders: List of PyTorch data loaders
        optimizer: PyTorch optimizer
        loggers: List of loggers

    Returns: None

    """

    """
    wandb.init()
        作用: 初始化一个 wandb 实验（run），启动与 wandb 服务器的连接，用于记录实验的日志、指标和配置。
        细节:
        wandb 是 Weights & Biases 的 Python 库，wandb.init() 是开始跟踪实验的入口。
        返回一个 Run 对象（这里赋值给 run），可以通过它手动记录数据或控制实验。
    
    run:
        作用: 接收 wandb.init() 的返回值，一个 Run 对象。
        细节:
        run 可用于后续操作，例如手动记录日志（run.log()）或结束实验（run.finish()）。
        如果不显式使用 run，wandb 也会通过全局状态跟踪实验。
    """
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"  # 通过设置代理，可以在线同步 Wandb 网站
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"  # 通过设置代理，可以在线同步 Wandb 网站
    # run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.name, config=cfg)

    run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.name, config=cfg, mode="offline")

    num_splits = len(loggers)  # 3：train, val, test
    full_epoch_times = []  # 记录每个 epoch 的运行时间
    perf = [[] for _ in range(num_splits-1)]  # 仅记录 train 和 val 的日志信息
    ckpt_dir = osp.join(cfg.run_dir, "ckpt/")  # results/xxx/ckpt/

    """
    学习率调度器，用于在训练神经网络时动态调整优化器的学习率。
    OneCycleLR 基于 "1Cycle" 学习率策略（One Cycle Policy），旨在通过在一个训练周期内动态调整学习率来加速收敛并提高模型性能。
    它的核心思想是：学习率从一个较低值开始，逐渐增加到最大值（max_lr），然后再逐渐下降到一个较低值，形成一个“上升-下降”的周期。
    
    optimizer: 传入的优化器对象（如 torch.optim.Adam 或 torch.optim.AdamW），调度器会根据策略调整该优化器的学习率。
    len(loaders[0]): 数据加载器（如训练集 DataLoader）中每个 epoch 的批次数量。
    cfg.batch_accumulation: 梯度累积的步数。如果每次迭代不立即更新权重，而是累积多次梯度后再更新，则需要除以这个值来计算实际的更新次数。
    pct_start: 表示学习率从初始值上升到 max_lr 所占的步数比例。通常是一个 0 到 1 之间的浮点数，例如 0.3，表示前 30% 的步数用于学习率从低值上升到 max_lr。
    """
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

    if cfg.useSWA:  # 是否使用 SWA
        swa_model = swa_utils.AveragedModel(model)

    for cur_epoch in range(cfg.optim.max_epoch):
        start_time = time.perf_counter()  # perf_counter 是 time 模块的一个高精度计时函数
        
        train_epoch(loggers[0], loaders[0], model, optimizer, cfg.batch_accumulation, scheduler, cur_epoch=cur_epoch)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        eval_epoch(loggers[1], loaders[1], model)
        perf[1].append(loggers[1].write_epoch(cur_epoch))

        # 记录 训练、验证 运行时间
        full_epoch_times.append(time.perf_counter() - start_time)    
        # 记录日志
        run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.     
        best_epoch = int(np.array([vp['MAE'] for vp in perf[1]]).argmin())  # argmin() 用于返回数组中最小值的索引
        best_train = f"train_MAE: {perf[0][best_epoch]['MAE']:.4f}"
        best_val = f"val_MAE: {perf[1][best_epoch]['MAE']:.4f}"
        bstats = {"best/epoch": best_epoch}
        for i, s in enumerate(['train', 'val']): 
            bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
            bstats[f"best/{s}_MAE"] = perf[i][best_epoch]['MAE']
        logging.info(bstats)  # 在控制台打印统计信息。
        run.log(bstats, step=cur_epoch)  # 将统计信息上传到 wandb。

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

    """
    第一次训练解释后，出现  MemoryError，判断是内存空间不够：
    1. 减少batch size;
    2. 减少 workers;
    3. 使用 gc.collect() 释放内存。(但是不知道的是否会将 loggers、loaders、model 都释放掉，需要测试)
    """
    # gc.collect()

    if cfg.useSWA:  # 训练结束时调用，专门为 SWA 模型更新批归一化（Batch Normalization, BN）层。
        print("Updating BNs for Stochastic Weight Averaging")
        device = swa_model.parameters().__next__().device
        swa_utils.update_bn(loaders[0], swa_model, device=device)
        torch.save(swa_model.state_dict(), osp.join(ckpt_dir, 'swa.ckpt'))

    """
    测试阶段
    """
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

    if cfg.useSWA:  # SWA 测试
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
    optimizer.zero_grad()  # 在每一轮开始前，清空优化器的梯度
    cur_epoch = kwargs.get('cur_epoch', 0)  # 默认值为 0

    """
    total=len(loader): 设置进度条的总长度为数据加载器的批次数量。
    ncols=50: 设置进度条宽度为 50 个字符，控制显示长度。
    """
    for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
        time_start = time.time()
        batch.to("cuda:0")

        pred, true = model(batch)  # 模型的返回值：预测值，标签值
            
        MAE, MSE = compute_loss(pred, true)

        if cfg.loss == "MAE":
            loss = MAE
        elif cfg.loss == "MSE":
            loss = MSE
        else:
            raise Exception("Loss not implemented")

        """
        loss.mean().backward()
        作用: 计算损失的梯度并进行反向传播。
        细节:
        loss 是一个张量，可能具有多个元素（例如每个样本的损失）。
        .mean(): 如果 loss 不是标量，取平均值将其转换为标量（PyTorch 要求反向传播的损失是标量）。
        .backward(): 执行反向传播，计算模型参数的梯度，存储在参数的 .grad 属性中。
        """
        loss.mean().backward()

        """
        根据梯度累积策略更新模型参数，并调整学习率。
        1. optimizer.step(): 使用累积的梯度更新模型参数。
        2. scheduler.step(): 根据学习率调度器（如 OneCycleLR）调整优化器的学习率。
        3. optimizer.zero_grad(): 清零梯度，为下一轮计算准备。
        """
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            optimizer.step()
            # 判断是否更新学习率
            if not cfg.useSWA or cfg.useSWA and cur_epoch + cfg.swa_epochs < cfg.optim.max_epoch:
                scheduler.step()
            optimizer.zero_grad()

        # .detach(): 去除梯度，将张量从计算图中分离，防止日志记录影响梯度计算。
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
