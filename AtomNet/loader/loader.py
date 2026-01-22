import torch
from dataset.figshare_dataset import Figshare_Dataset
from dataset.utils import compute_knn
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
from jarvis_utils import *
from datetime import datetime


def create_loader():
    if cfg.dataset.name == "jarvis" or cfg.dataset.name == "megnet":
        if cfg.dataset.name == "jarvis":
            cfg.dataset.name = "dft_3d_2021"

        seed = 123
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
            """ 用于 autodl 等无法加载 jdata 数据集的云服务器 
            if cfg.dataset.name == "dft_3d_2021":
                jdata_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/jarvis/original/')
            # ① 论文：读取数据集
            data = jdata(cfg.dataset.name, store_dir=jdata_path)
            """
            # ① 论文：读取数据集
            data = jdata(cfg.dataset.name)

            # ② 只选含铁、钴、镍元素的材料
            # data = getDataById(cfg.dataset.name)  # 自己写的 utils

            print("数据集的长度：", len(data))
            dat = []
            all_targets = []
            # 数据异常值筛选
            for i in data:
                if isinstance(i[target], list):  # 如果i[target]是列表
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

        radius = cfg.radius  # 邻域的截止半径
        # "dft_3d_2021_5_-1_formation_energy_peratom_123"
        prefix = cfg.dataset.name+"_"+str(radius)+"_"+str(cfg.max_neighbours)+"_"+target+"_"+str(seed)
        print("dat_train 的长度", len(dat_train))
        print("dat_val 的长度", len(dat_val))
        print("dat_test 的长度", len(dat_test))
        dataset_train = Figshare_Dataset(root=cfg.dataset_path, data=dat_train, targets=targets_train, radius=radius,
                                         max_neigh=cfg.max_neighbours, name=prefix+"_train", atom_init=cfg.atom_init)
        print("打印 dataset_train.data: ", dataset_train.data)  # Data 对象
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


def create_inference_loader():
    """
    data 是 json 文件，由 atoms 字典组成的列表，每个 atoms 字典代表一个晶体结构，要求包含以下属性：
    -----------------
    lattice_mat: 晶格矩阵，3x3 的 numpy 数组
    coords: 坐标，nx3 的 numpy 数组
    elements: 元素，n 个元素的列表
    "cartesian": 是否使用笛卡尔坐标
    "props": [] 用空列表即可，用不到
    """
    assert open(cfg.inference_data_path, "r").readable(), f"{cfg.inference_data_path} is not readable"
    data = json.load(open(cfg.inference_data_path, "r"))
    print("The length of the inference dataset: ", len(data))

    # Tags are not needed for reasoning
    all_targets = [0.] * len(data)

    if len(data):
        dataset = Figshare_Dataset(root=cfg.inference_path, data=data, targets=all_targets, radius=cfg.radius,
                                   max_neigh=cfg.max_neighbours, name=f"inference_{datetime.now().strftime('%Y%m%d%H%M')}",
                                   atom_init=cfg.atom_init)
    else:
        raise Exception("Datasets are not available!")

    return DataLoader(dataset, batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers, pin_memory=True)


"""
数据集划分训练集、验证集和测试集，返回数据集的下标
"""
def create_train_val_test(data, val_ratio=0.1, test_ratio=0.1, seed=123):
    ids = list(np.arange(len(data)))
    n = len(data)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    random.seed(seed)
    random.shuffle(ids)
    ids_train = ids[:n_train]
    ids_val = ids[-(n_val + n_test): -n_test]  # 从倒数 (n_val + n_test) 到倒数 n_test 的部分作为验证集。
    ids_test = ids[-n_test:]
    return ids_train, ids_val, ids_test