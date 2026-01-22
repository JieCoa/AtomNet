from __future__ import print_function, division
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.config import cfg
from tqdm.auto import tqdm
from jarvis.core.specie import get_node_attributes
from jarvis.core.atoms import Atoms
from torch_geometric.data import Data, Batch
from dataset.utils import radius_graph_pbc
from sklearn.preprocessing import StandardScaler
import torch
import roma
import numpy as np
import json
import os, sys


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)  # 原子序数集合
        self._embedding = {}  # 原子序数作为 key，保存对应的原子初始特征为 value

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        # 通过 .items() 读取字典中每一个键值对
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class Figshare_Dataset(InMemoryDataset):
    def __init__(self, root, data, targets, transform=None, pre_transform=None, name="jarvis", radius=5.0, max_neigh=-1,
                 augment=False,  atom_init='atom_number'):
        """
        root: 根目录，告诉基类数据集存储位置。 root = "./dataset/jarvis/"
        InMemoryDataset 会根据 root 设置 raw_dir（原始数据目录）和 processed_dir（处理后数据目录）。
         - 在 InMemoryDataset 的初始化中，processed_paths 是通过 processed_file_names 属性计算得到的。
         - 默认行为:
            如果子类没有重写 processed_file_names，InMemoryDataset 会假设默认文件名为 "data.pt"。
            因此，self.processed_paths[0] 通常是 root/processed/data.pt。
        """
        if atom_init != 'atom_number':
            self.json_file = atom_init + '.json'
            self.name = name + '_' + self.json_file  # "dft_3d_2021_5_25_formation_energy_peratom_123_train" + '_' + json_file
            atom_init_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json/', self.json_file)

            assert os.path.exists(atom_init_file), f'{self.json_file} does not exist!'
        else:
            self.name = name  # "dft_3d_2021_5_25_formation_energy_peratom_123_train"

        if cfg.useElectronegativity:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json/electronegativity(sanderson).json'), 'r') as f:
                self.eneg_dict = json.load(f)
            if cfg.normalizedElectronegativity:
                self.name += '_normalized_electronegativity'
            else:
                self.name += '_electronegativity'

        self.data = data
        self.targets = targets

        self.radius = radius
        self.max_neigh = max_neigh if max_neigh > 0 else None
        self.augment = augment

        if atom_init != 'atom_number':
            self.ari = AtomCustomJSONInitializer(atom_init_file)

        super(Figshare_Dataset, self).__init__(root, transform, pre_transform)
        """
        若 jarvis/processed 中有同名(self.name) 数据存在，则不会执行下面的 process 方法
        """
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.name + ".pt"

    def download(self):
        pass
    

    def get(self, idx):
        data = super().get(idx)
        
        if self.augment:
            data = self.augment_data(data)
        
        return data

    def augment_data(self, data):  # 旋转SO(3)数据增强
        R = roma.utils.random_rotmat(size=1, device=data.x.device).squeeze(0)    
        data.cart_dir = data.cart_dir @ R
        data.cell = data.cell @ R

        return data

    def process(self):
        """
        1. 在 PyTorch Geometric 的 InMemoryDataset 基类中，process() 方法的调用并不是直接在 __init__ 中显式执行的，
           而是通过属性访问的懒加载机制（lazy loading）触发的。具体来说，InMemoryDataset 在需要访问 self.data 或 self.slices 时，
           会检查处理后的文件是否存在，如果不存在，则调用 process() 方法生成文件。这种机制依赖于 Python 的 @property 装饰器和内部逻辑。
        2. self.data 和 self.slices 的懒加载
         - InMemoryDataset 使用 @property 装饰器定义了 data 和 slices 属性。当你访问 self.data 或 self.slices 时，会触发内部的加载逻辑。
        """
        data_list = []
        max_eneg = 0
        scaler = StandardScaler()
        for i, (ddat, target) in tqdm(enumerate(zip(self.data, self.targets)), total=len(self.data)):
            """
            每个 ddat 就是一个晶体材料
            将 ddat["atoms"] 字典转换为 jarvis.core.atoms.Atoms 对象；
            输出 structure 包含：
                lattice.matrix: 晶格矩阵（[[1, 0, 0], [0, 1, 0], [0, 0, 1]]）。
                cart_coords: 笛卡尔坐标（[[0, 0, 0], [0.5, 0.5, 0.5]]）。
                elements: 元素列表（["H", "O"]）。
            """
            structure = Atoms.from_dict(ddat["atoms"])

            """
            structure.elements: ["H", "O"]。
            get_node_attributes: 获取原子序数，例如：
            "H" → 1  "O" → 8   列表推导式: [1, 8]。
            .squeeze(-1) 移除张量中指定维度（这里是最后一个维度，索引为 -1）上大小为 1 的维度, 因为这是一个一维张量，没有额外的维度为 1，
            因此 .squeeze(-1) 不会改变形状，仍然是 (3,)。
            
            【注意】如果 get_node_attributes 返回的是形状为 (1,) 的张量（例如 tensor([1])），列表推导式结果可能是 [[1], [8], [26]]，
            转为张量后是 (3, 1)，那么 .squeeze(-1) 会将其变为 (3,)。但根据上下文和 jarvis 的常见用法，这里假设返回的是标量，所以直接就是 (3,)。
            
            作用: atomic_numbers 是节点特征 (x)，表示每个原子的原子序数。
            """
            if hasattr(self, "json_file"):  # 判断是否有 self.json_file 这个属性，若有，则使用了原子特征初始化 json 文件
                # 下面 5 行是自己改的
                atomic_numbers = [get_node_attributes(s, atom_features="atomic_number") for s in structure.elements]
                # print("atomic_numbers: ", atomic_numbers)
                atom_features = np.vstack([self.ari.get_atom_fea(i[0]) for i in atomic_numbers])
                # print("打印atom_features: ", atom_features)
                atom_features = torch.tensor(atom_features)

                data = Data(x=atom_features, y=target)
            else:  # 使用 '原子序数' 作为原子特征
                atomic_numbers = torch.tensor([get_node_attributes(s, atom_features="atomic_number") for s in structure.elements]).squeeze(-1)

                data = Data(x=atomic_numbers, y=target)

            # data.eneg = torch.tensor([element(i[0]).electronegativity(scale='sanderson') for i in atomic_numbers])
            data.pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
            data.cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

            """
            这里的 [[True, True, True]] 是一个形状为 (1, 3) 的张量，表示一个晶体的三个空间方向（x, y, z）都具有周期性。
            具体含义：
                [True, True, True]：x 方向周期性，y 方向周期性，z 方向周期性；
                外层的 [ ] 是批次维度（batch dimension），表示这是单个图的 PBC 设置。
            """
            data.pbc = torch.tensor([[True, True, True]])
            data.natoms = torch.tensor([data.x.shape[0]])  # 节点数

            # 形状: (3, 3) → 形状: (1, 3, 3)
            data.cell = data.cell.unsqueeze(0)  # 增加一个维度，[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]。

            # Compute PBC
            batch = Batch.from_data_list([data])

            edge_index, _, _, cart_vector = radius_graph_pbc(batch, self.radius, self.max_neigh)
            
            data.cart_dist = torch.norm(cart_vector, p=2, dim=-1)  # 计算原子之间的距离
            data.cart_dir = torch.nn.functional.normalize(cart_vector, p=2, dim=-1)  # 计算原子之间的方向向量

            # 每个晶体中，边的数量，edge_index：形状 (2, M)。
            # data.nedges = torch.tensor(edge_index.shape[-1])

            data.edge_index = edge_index
            if len(edge_index[0]) == 0:  # 如果不存在边，则剔除该数据
                continue

            # --- 边特征：原子对的电负性差值的绝对值（）---
            if cfg.useElectronegativity:
                diffs = []
                for i in range(edge_index.shape[1]):  # 对每列 i
                    e1 = self.eneg_dict[atomic_numbers[edge_index[0][i]][0] - 1]  # 原子序数 - 1
                    e2 = self.eneg_dict[atomic_numbers[edge_index[1][i]][0] - 1]
                    diff = abs(e1 - e2)
                    diffs.append(diff)
                if cfg.normalizedElectronegativity and len(diffs) > 0:  # scaler 要求输入数据是二维的
                    diffs = scaler.fit_transform(np.array(diffs).reshape(-1, 1))
                    data.eneg = torch.tensor(diffs, dtype=torch.float32)
                else:
                    data.eneg = torch.tensor(diffs, dtype=torch.float32).unsqueeze(-1)
                # if diffs:
                #     max_eneg = max(max_eneg, max(diffs))

            # --- 边特征：原子对的电负性差值的绝对值（）---
            # print("atomic_numbers: ", atomic_numbers)
            # print("data.eneg: ", data.eneg)
            # sys.exit("手动终止程序执行！")

            delattr(data, "pbc")
            data_list.append(data)

        # print("max_eneg: ", max_eneg)
        # sys.exit("手动终止程序执行！")
        # print(f"有效数据 {len(data_list)} 个。")
        #
        # """ 绘制边长统计柱状图 & KDE & 标识 length=1.0 的统计值 """
        # from plot_edge import collect_edge_lengths, plot_hist_and_annotate
        # all_dists = collect_edge_lengths(data_list, attr_name='cart_dist')
        # plot_hist_and_annotate(all_dists, n_bins=50, value_to_mark=1.0, title='Edge lengths')
        # print("绘制完成：边长统计柱状图。", all_dists)
        # sys.exit("手动终止程序执行！")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
