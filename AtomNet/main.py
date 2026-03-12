import os

import torch
import torch.nn.functional as F
import logging
import argparse
import pickle
from tqdm import tqdm
from logger.logger import create_logger
from loader.loader import create_loader, create_inference_loader
from models.master import create_model
from train.train import train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.logger import set_printing
import re
from ig_framework import IG_metric
import time
import pandas as pd


def inference(model, loader):
    """
    Run inference using the trained model and save the results.
    """
    import numpy as np
    import json

    model.eval()
    inference_list = []
    time_start = time.time()

    with torch.no_grad():
        for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
            batch.to("cuda:0")
            pred, _ = model(batch)
            inference_list.append(pred.detach().cpu().numpy())

    logging.info(
        f"(Inference time : {np.mean(time.time() - time_start):.1f}s) | "
        f"Inference results : {inference_list}\t"
    )

    inference_list_json = [pred.tolist() for pred in inference_list]
    logging.info(f"Inference results : {inference_list_json}\t")

    with open("dataset/inference/inference_results.json", "w") as f:
        json.dump(inference_list_json, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for the experiment')
    parser.add_argument('--name', type=str, default="Test", help="name of the Wandb experiment" )  # jarvis_dft_3D_formation_energy_peratom
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--batch_accumulation", type=int, default=1, help="Batch Accumulation")  # It is related to the learning rate scheduler
    parser.add_argument("--dataset", type=str, default="megnet", choices=["megnet", "jarvis"], help="Dataset name. Available: jarvis, megnet")
    parser.add_argument("--dataset_path", type=str, default="./dataset/jarvis/")
    parser.add_argument("--inference", action="store_true", help="Inference")
    parser.add_argument("--inference_path", type=str, default="./dataset/inference/")
    parser.add_argument("--inference_data_path", type=str, default="./dataset/inference/inference_data.json")
    # parser.add_argument("--inference_output", type=str, default="./inference.pkl", help="Path to the inference output")
    parser.add_argument("--figshare_target", type=str, default="gap pbe", help="Figshare dataset target")  # mbj_bandgap, optb88vdw_bandgap, ...
    parser.add_argument("--wandb_project", type=str, default="please fill your project name", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="please fill your entity name",
                        help="Name of the wandb entity")  # your own wandb account name
    parser.add_argument("--loss", type=str, default="MAE", help="Loss function")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup", type=float, default=0.01, help="Warmup")
    parser.add_argument('--model', type=str, default="AtomNet", help="Model Name")
    parser.add_argument("--max_neighbours", type=int, default=-1, help="Max neighbours")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius for the Radius Graph Neighbourhood")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dim_in", type=int, default=256, help="Input dimension")
    parser.add_argument("--dim_rbf", type=int, default=64, help="Number of RBF")
    parser.add_argument("--invariant", action="store_true", help="Rotation Invariant model")
    parser.add_argument("--disable_envelope", action="store_false", help="Disable envelope")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")  # Process

    parser.add_argument("--atom_init", type=str, default="atom_number", choices=["atom_number", "atom_features(116d)_update01"],
                        help="If use the atom initial file")
    parser.add_argument("--usePolynomial", type=int, default=3, help="Use polynomial features for the edge features")
    parser.add_argument("--disableUpdateEdge", action='store_true', help='Stop updating edge features')
    parser.add_argument("--limitedUpdateEdge", type=int, default=0, help='Limit the number of edge feature updates')
    parser.add_argument("--newEnvelope", action='store_false', help='Use new envelope')
    parser.add_argument("--envelope_type", type=str, default='cubic', choices=["cubic", "simply"], help='Weight function for the envelope')  # 'simply'
    parser.add_argument("--normalized_Polynomial", action='store_true', help='Use normalized Polynomial features')
    parser.add_argument("--useSWA", action='store_true', help="Use Stochastic Weight Averaging")
    parser.add_argument("--swa_epochs", type=int, default=20, help="Number of epochs for SWA")
    parser.add_argument("--useElectronegativity", action='store_true', help='Use Electronegativity features for the edge features')
    parser.add_argument("--normalizedElectronegativity", action='store_true', help='Use normalized Electronegativity features for the edge features')
    parser.add_argument("--onlyAtomFeaUpdate", action='store_true', help='Only update atom features in node-wise')
    parser.add_argument("--ig", action='store_true', help='Use IG for feature importance')
    parser.add_argument("--electronegativity_type", type=str, default=None,
                        choices=[None, "newRBF", "newRBF02", "newRBF03", "newRBF04", "newRBF05"],
                        help="Number of epochs for IG (default: None)")

    assert torch.cuda.is_available(), "CUDA is unavailable. Please check if the GPU is occupied or unrecognized!"

    torch.cuda.empty_cache()

    set_cfg(cfg)

    args = parser.parse_args()
    cfg.seed = args.seed
    cfg.dataset.task_type = "regression"
    cfg.batch = args.batch
    cfg.batch_accumulation = args.batch_accumulation
    cfg.dataset.name = args.dataset  # jarvis
    cfg.dataset_path = args.dataset_path
    cfg.inference_path = args.inference_path
    cfg.inference_data_path = args.inference_data_path
    cfg.useProcessedData = True  # inference dataset
    cfg.figshare_target = args.figshare_target
    cfg.wandb_project = args.wandb_project
    cfg.wandb_entity = args.wandb_entity
    cfg.loss = args.loss
    cfg.optim.max_epoch = args.epochs
    cfg.lr = args.lr
    cfg.warmup = args.warmup
    cfg.model = args.model
    # We found that when the number of neighbors was not restricted, the model's performance improved.
    cfg.max_neighbours = args.max_neighbours
    cfg.radius = args.radius
    cfg.num_layers = args.num_layers
    cfg.dim_in = args.dim_in
    cfg.dim_rbf = args.dim_rbf
    cfg.invariant = args.invariant
    cfg.envelope = args.disable_envelope  # True
    cfg.workers = args.workers

    cfg.atom_init = args.atom_init
    cfg.usePolynomial = args.usePolynomial
    cfg.disableUpdateEdge = args.disableUpdateEdge
    cfg.limitedUpdateEdge = args.limitedUpdateEdge
    cfg.newEnvelope = args.newEnvelope
    cfg.normalized_Polynomial = args.normalized_Polynomial
    cfg.useSWA = args.useSWA
    cfg.swa_epochs = args.swa_epochs
    cfg.useElectronegativity = args.useElectronegativity
    cfg.normalizedElectronegativity = args.normalizedElectronegativity
    cfg.onlyAtomFeaUpdate = args.onlyAtomFeaUpdate

    cfg.NormLayer = 'BN' if cfg.figshare_target == 'mbj_bandgap' else 'LN'
    cfg.eneg_type = args.electronegativity_type if cfg.useElectronegativity and not cfg.normalizedElectronegativity else None
    cfg.envelope_type = args.envelope_type if cfg.newEnvelope else None

    # 根据参数生成对应的 name
    parts = [f"seed({cfg.seed})_({cfg.batch}batch"]
    if cfg.envelope_type:
        parts.append(str(cfg.envelope_type))
    if cfg.eneg_type:
        parts.append(str(cfg.eneg_type))
    if cfg.useElectronegativity:
        parts.append("normed_eneg" if cfg.normalizedElectronegativity else "eneg")
    if cfg.max_neighbours != -1:
        parts.append(f"max_neighbours({cfg.max_neighbours})")

    cfg.name = "_".join(parts) + f"_{cfg.atom_init})" + args.name

    cfg.run_dir = "results/" + cfg.name

    torch.set_num_threads(args.threads)

    set_printing()

    #Seed
    seed_everything(cfg.seed)

    logging.info(f"RBF type for Electronegativity: {cfg.eneg_type}")
    logging.info(f"Envelope type: {cfg.envelope_type}")
    logging.info(f"The number of neighbors is set to {cfg.max_neighbours}.")
    logging.info(f"Name of the wandb record: {cfg.name}")
    logging.info(f"Target property: {cfg.figshare_target}")
    logging.info(f"Experiment will be saved at: {cfg.run_dir}")

    if args.inference:  # Inference uses an additional incoming dataset (requiring a list of dictionaries containing the specified key)
        inference_loader = create_inference_loader(use_processed_data=cfg.useProcessedData)
        logging.info(f"Use processed dataset for Inference: {cfg.useProcessedData}.")
    else:
        loaders = create_loader()

    model = create_model()

    logging.info(model)
    cfg.params_count = params_count(model)
    logging.info(f"Number of parameters: {cfg.params_count}")

    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.01)

    loggers = create_logger()

    if args.inference:
        ckpt_path = f'{cfg.run_dir}/ckpt/best.ckpt'
        assert os.path.exists(ckpt_path), "The model parameter file does not exist! ❌"

        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state"])
        # cfg.inference_output = args.inference_output

        inference(model, inference_loader, cfg.useProcessedData)

    elif args.ig:  # Feature importance ranking
        logging.info("📊 Start computing feature importance...")
        IG_metric(model, loaders[0])

    else:
        train(model, loaders, optimizer, loggers)
