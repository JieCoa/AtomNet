import logging
import torch
from dataset.figshare_dataset import Figshare_Dataset
from dataset.inference_input import load_inference_records
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader
from jarvis.db.figshare import data as jdata
import random
import os.path as osp
import numpy as np
import math
import pickle as pk
import gc
import pandas as pd
import tempfile


def create_loader(seed=123):
    if cfg.dataset.name == "jarvis" or cfg.dataset.name == "megnet":
        if cfg.dataset.name == "jarvis":
            cfg.dataset.name = "dft_3d_2021"

        target = cfg.figshare_target
        if cfg.figshare_target in ["shear modulus", "bulk modulus"] and cfg.dataset.name == "megnet":
            target = cfg.figshare_target
            if cfg.figshare_target == "bulk modulus":
                try:
                    data_train = pk.load(open("./dataset/megnet/bulk_megnet_train.pkl", "rb"))
                    data_val = pk.load(open("./dataset/megnet/bulk_megnet_val.pkl", "rb"))
                    data_test = pk.load(open("./dataset/megnet/bulk_megnet_test.pkl", "rb"))
                except:
                    raise Exception("Bulk modulus dataset not found, please download it from https://figshare.com/projects/Bulk_and_shear_datasets/165430")
            elif cfg.figshare_target == "shear modulus":
                try:
                    data_train = pk.load(open("./dataset/megnet/shear_megnet_train.pkl", "rb"))
                    data_val = pk.load(open("./dataset/megnet/shear_megnet_val.pkl", "rb"))
                    data_test = pk.load(open("./dataset/megnet/shear_megnet_test.pkl", "rb"))
                except:
                    raise Exception("Shear modulus dataset not found, please download it from https://figshare.com/projects/Bulk_and_shear_datasets/165430")
            
            targets_train = []
            dat_train = []
            targets_val = []
            dat_val = []
            targets_test = []
            dat_test = []
            for split, datalist, targets in zip([data_train, data_val, data_test], 
                                                [dat_train, dat_val, dat_test],
                                                [targets_train, targets_val, targets_test]):
                for i in split:
                    if (
                        i[target] is not None
                        and i[target] != "na"
                        and not math.isnan(i[target])
                    ):
                        datalist.append(i)
                        targets.append(i[target])

        else:
            data = jdata(cfg.dataset.name)

            print("The size of dataset：", len(data))
            dat = []
            all_targets = []

            for i in data:
                if isinstance(i[target], list):
                    all_targets.append(torch.tensor(i[target]))
                    dat.append(i)
                elif i[target] is not None and i[target] != "na" and not math.isnan(i[target]):
                    dat.append(i)
                    all_targets.append(i[target])
            
            ids_train, ids_val, ids_test = create_train_val_test(dat, seed=seed)

            dat_train = [dat[i] for i in ids_train]
            dat_val = [dat[i] for i in ids_val]
            dat_test = [dat[i] for i in ids_test]
            targets_train = [all_targets[i] for i in ids_train]
            targets_val = [all_targets[i] for i in ids_val]
            targets_test = [all_targets[i] for i in ids_test]

            del data
            del dat

        radius = cfg.radius
        # "dft_3d_2021_5_-1_formation_energy_peratom_123"
        prefix = cfg.dataset.name+"_"+str(radius)+"_"+str(cfg.max_neighbours)+"_"+target+"_"+str(seed)

        dataset_train = Figshare_Dataset(root=cfg.dataset_path, data=dat_train, targets=targets_train, radius=radius,
                                         max_neigh=cfg.max_neighbours, name=prefix+"_train", atom_init=cfg.atom_init)
        dataset_val = Figshare_Dataset(root=cfg.dataset_path, data=dat_val, targets=targets_val, radius=radius,
                                       max_neigh=cfg.max_neighbours, name=prefix+"_val", atom_init=cfg.atom_init)
        dataset_test = Figshare_Dataset(root=cfg.dataset_path, data=dat_test, targets=targets_test, radius=radius,
                                        max_neigh=cfg.max_neighbours, name=prefix+"_test", atom_init=cfg.atom_init)

        gc.collect()

    else:
        raise Exception("Dataset not implemented")
    
    loaders = [
        DataLoader(dataset_train,
                   batch_size=cfg.batch,
                   persistent_workers=True,
                   shuffle=True,
                   num_workers=cfg.workers,
                   pin_memory=True
                   ),
        DataLoader(dataset_val,
                   batch_size=cfg.batch,
                   persistent_workers=True,
                   shuffle=False,
                   num_workers=cfg.workers,
                   pin_memory=True
                   ),
        DataLoader(dataset_test,
                   batch_size=cfg.batch,
                   persistent_workers=False,
                   shuffle=False,
                   num_workers=cfg.workers,
                   pin_memory=True
                   )
    ]
    
    return loaders


def create_inference_loader(use_processed_data=False):
    """
    构建推理 loader，返回 (DataLoader, ids)。

    输入可以是 jarvis 风格 json（列表元素包含 "atoms"）、单个 CIF 文件，或包含多个
    CIF 文件的目录；三者会被 dataset.inference_input 归一化成同一种结构。

    atoms 字典需包含：lattice_mat (3x3)、coords (n x 3)、elements (n)；
    "cartesian" 决定坐标含义（CIF 转出的是分数坐标），"props" 可为空列表。
    ids 与输入记录一一对应，推理结果通过 dataset 中保留的原始下标映射回 ids。
    """
    if use_processed_data:  # Use the test dataset for reasoning
        print("Loading processed test data...")
        return create_loader()[2], None

    print(f"Loading inference data from {cfg.inference_data_path} ...")
    records = load_inference_records(cfg.inference_data_path, primitive=getattr(cfg, "cif_primitive", False))
    print("The length of the inference dataset: ", len(records))

    # Tags are not needed for reasoning
    ids = [record["id"] for record in records]
    all_targets = [0.] * len(records)

    if not records:
        raise Exception("Datasets are not available!")

    # 推理数据只用于本次预测，不需要持久化预处理结果。
    # 使用临时目录作为 InMemoryDataset 的 root，dataset 初始化后会删除临时文件。
    with tempfile.TemporaryDirectory(prefix="atomnet_inference_") as tmp_root:
        dataset = Figshare_Dataset(root=tmp_root, data=records, targets=all_targets, radius=cfg.radius,
                                   max_neigh=cfg.max_neighbours, name="inference",
                                   atom_init=cfg.atom_init, ids=ids)

    if len(dataset) < len(records):
        logging.warning(f"{len(records) - len(dataset)} 个结构未生成任何边，已从推理数据中跳过")

    return DataLoader(dataset, batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers, pin_memory=True), ids


def create_train_val_test(data, val_ratio=0.1, test_ratio=0.1, seed=123):
    ids = list(np.arange(len(data)))
    n = len(data)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    random.seed(seed)
    random.shuffle(ids)
    ids_train = ids[:n_train]
    ids_val = ids[-(n_val + n_test): -n_test]
    ids_test = ids[-n_test:]
    return ids_train, ids_val, ids_test