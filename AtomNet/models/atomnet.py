import logging
import math
import torch
import torch_geometric.nn as pyg_nn
import torch.nn as nn
from torch import Tensor
from torch_geometric.graphgym.config import cfg
from torch_scatter import scatter
from models.utils import ExpNormalSmearing, CosineCutoff, RBF2Electronegativity
import re
import sys


class AtomNet(torch.nn.Module):
    def __init__(self, 
        dim_in: int, 
        dim_rbf: int, 
        num_layers: int,
        radius: float = 5.0,
        invariant: bool = False,
        use_envelope: bool = True,
        atom_init: str = "atom_number",
        ):
        super().__init__()
        print(f"打印 atom_init: {atom_init}")
        print(f"是否使用 Envelope: {use_envelope}")

        self.encoder = Encoder(dim_in, dim_rbf=dim_rbf, radius=radius, invariant=invariant, atom_init=atom_init)
        self.dim_in = dim_in  # 256

        layers = []
        for i in range(num_layers):  # for _ in range(num_layers):
            layers.append(AtomNet_layer(
                idx=i,  # 新添加参数：根据层数，进行有限次特征更新
                dim_in=dim_in,
                use_envelope=use_envelope,
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.head = Readout(dim_in)
        
    def forward(self, batch):
        batch = self.encoder(batch)

        for layer in self.layers:
            batch = layer(batch)
        
        pred, true = self.head(batch)
        
        return pred, true


class Encoder(torch.nn.Module):  
    def __init__(
        self,
        dim_in: int,   # 256
        dim_rbf: int,  # 64
        radius: float = 5.0,
        invariant: bool = False,
        atom_init: str = "atom_number",
    ):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.invariant = invariant
        self.atom_init = atom_init

        """
        1. Embedding：参数119表示嵌入字典的大小，即有119个不同的原子类型。参数self.dim_in*2表示每个嵌入向量的维度，
        即每个原子类型将被映射到一个维度为self.dim_in*2的向量。
        2. 使用Xavier均匀分布对嵌入层的权重进行初始化。Xavier初始化是一种权重初始化方法，旨在保持输入和输出的方差一致，
        以帮助梯度更平稳地传播，从而避免训练过程中的梯度消失或爆炸问题。
        """
        if re.search(r'\d+', atom_init):  # 提取 atom_init 中的数字
            number = int(re.search(r'\d+', atom_init).group())
            self.embedding = nn.Linear(number, self.dim_in * 2)

        elif atom_init == 'atom_number':
            self.embedding = nn.Embedding(119, self.dim_in * 2)  # 119 → 512

        elif atom_init == 'cgcnn':
            self.embedding = nn.Linear(92, self.dim_in * 2)  # 92 → 512

        else:  # atom_features(136d)等
            assert "未识别到对应的节点特征维度！"
            # self.embedding = nn.Linear(136, self.dim_in * 2)  # 136 → 512
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        """
        nn.Parameter 是 PyTorch 中用于定义可训练参数的类，表示这个参数会在训练过程中被优化。
         - inplace=True 表示激活函数会直接在输入张量上进行操作，节省内存。
        """
        self.bias = nn.Parameter(torch.zeros(self.dim_in*2))
        self.activation = nn.SiLU(inplace=True)

        # 原子编码器
        self.encoder_atom = nn.Sequential(self.activation,
                                          pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                          self.activation,  # 5-①
                                          )
        if self.invariant:
            dim_edge = dim_rbf
        elif cfg.usePolynomial > 0:  # 多项式特征衍生(1、2、3阶)
            dim_edge = dim_rbf + 3 + cfg.usePolynomial  # 64 + 3 + m
        else:
            dim_edge = dim_rbf + 3  # 64 + 3
            # dim_edge = dim_rbf + 3 + 3  # 额外增加 3 个维度的 angle 信息
        if cfg.useElectronegativity:
            if cfg.normalized_Polynomial or cfg.eneg_type is None:
                dim_edge += 1
            elif cfg.eneg_type == 'newRBF':
                dim_edge += 20  # newRBF
            elif cfg.eneg_type is not None:
                dim_edge += 64  # newRBF02 & newRBF03 & newRBF04 & newRBF05
            else:
                raise ValueError("未识别到对应的电负性差值类型！")


        self.encoder_edge = nn.Sequential(pyg_nn.Linear(dim_edge, self.dim_in*2),
                                        self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)

        self.rbf = ExpNormalSmearing(0.0, radius, dim_rbf, False)  # Initialize the RBF (radial basis function)

        if cfg.useElectronegativity and cfg.eneg_type is not None:  # newRBF & newRBF02 & newRBF03 & newRBF04 & newRBF05
            self.rbf_eneg = RBF2Electronegativity(cutoff_upper=2.14, type=cfg.eneg_type, trainable=False)

    def forward(self, batch):
        # print(f"batch.x.shape：{batch.x.shape}")
        if self.atom_init == 'atom_number':
            x = self.embedding(batch.x) + self.bias
        else:
            # 因为 json 中的列表是 Long 类型，而模型的参数矩阵类型是 float
            x = self.embedding(batch.x.float()) + self.bias

        batch.x = self.encoder_atom(x)

        if cfg.invariant:  # 默认 False
            batch.edge_attr = self.encoder_edge(self.rbf(batch.cart_dist))  # 将欧几里得距离传入 RBF
        elif cfg.usePolynomial > 0:  # 多项式特征衍生(1、2、3阶)
            dist = batch.cart_dist.unsqueeze(-1)
            if cfg.normalized_Polynomial:
                dist_min = dist.min(dim=0, keepdim=True).values
                dist_max = dist.max(dim=0, keepdim=True).values
                dist_normalized = (dist - dist_min) / (dist_max - dist_min + 1e-8)  # 避免分母为0
                polynomial = [dist_normalized ** i for i in range(1, cfg.usePolynomial + 1)]
            else:
                polynomial = [dist ** i for i in range(1, cfg.usePolynomial + 1)]

            polynomial = torch.cat(polynomial, dim=-1)

            if cfg.useElectronegativity:  # 是否使用原子对的电负性差值的绝对值作为边特征
                if cfg.eneg_type is None:
                    batch.edge_attr = self.encoder_edge(
                        torch.cat([self.rbf(batch.cart_dist), polynomial, batch.cart_dir, batch.eneg],
                                  dim=-1))
                else:
                    batch.edge_attr = self.encoder_edge(
                        torch.cat([self.rbf(batch.cart_dist), polynomial, batch.cart_dir, self.rbf_eneg(batch.eneg)],
                                  dim=-1))
            else:
                batch.edge_attr = self.encoder_edge(
                    torch.cat([self.rbf(batch.cart_dist), polynomial, batch.cart_dir], dim=-1))
        else:  # dim=-1 表示沿着最后一个维度进行拼接
            if cfg.useElectronegativity:
                if cfg.eneg_type is None:
                    batch.edge_attr = self.encoder_edge(
                        torch.cat([self.rbf(batch.cart_dist), batch.cart_dir, batch.eneg], dim=-1))
                else:
                    batch.edge_attr = self.encoder_edge(
                        torch.cat([self.rbf(batch.cart_dist), batch.cart_dir, self.rbf_eneg(batch.eneg)], dim=-1))
            else:
                # 1. 原始论文
                batch.edge_attr = self.encoder_edge(torch.cat([self.rbf(batch.cart_dist), batch.cart_dir], dim=-1))

        return batch

class AtomNet_layer(pyg_nn.conv.MessagePassing):
    """
    The message-passing layer used in the AtomNet architecture.
    Parameters:
        idx (int): range from 0 to len(num_layers)-1.
        dim_in (int): Dimension of the input node features.
        use_envelope (bool, optional): If True, applies an envelope function to the distances. Defaults to True.
    """
    
    def __init__(self,
        idx: int,
        dim_in: int, 
        use_envelope: bool = True
    ):
        super().__init__(node_dim=0)
        self.edge_batch = None  # 后续用于识别哪些边属于同一个 graph, 在 Transformer 中使用
        self.idx = idx  # 新增参数
        self.dim_in = dim_in
        self.activation = nn.SiLU(inplace=True)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.1)
        # self.softmax = nn.Softmax(dim=-1)

        if cfg.onlyAtomFeaUpdate:  # 节点特征更新只用原子特征，不用边特征
            self.MLP_aggr = nn.Sequential(
                pyg_nn.Linear(dim_in * 2, dim_in, bias=True),
                self.activation,
                pyg_nn.Linear(dim_in, dim_in, bias=True),
            )
        else:
            self.MLP_aggr = nn.Sequential(
                pyg_nn.Linear(dim_in*3, dim_in, bias=True),
                self.activation,
                pyg_nn.Linear(dim_in, dim_in, bias=True),
            )
        self.MLP_gate = nn.Sequential(
            pyg_nn.Linear(dim_in*3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )

        if cfg.NormLayer == 'BN':
            self.norm = nn.BatchNorm1d(dim_in)
        elif cfg.NormLayer == 'LN':
            self.norm = nn.LayerNorm(dim_in)
        else:
            raise ValueError("NormLayer must be 'BN' or 'LN'")

        self.norm2 = nn.BatchNorm1d(dim_in)
        self.use_envelope = use_envelope
        if self.use_envelope:
            self.envelope = CosineCutoff(0, cfg.radius, cfg.newEnvelope)

    def forward(self, batch):
        x, e, edge_index, dist = batch.x, batch.edge_attr, batch.edge_index, batch.cart_dist
        """
        x               : [n_nodes, dim_in]
        e               : [n_edges, dim_in], e.g. torch.Size([21382, 256])
        edge_index      : [2, n_edges] == [[邻居原子索引], [中心原子索引]]
        dist            : [n_edges]
        batch           : [n_nodes]
        """
        x_in = x
        e_in = e

        # 调用 self.propagate，触发消息传递流程（包括 message、aggregate 和 update）。
        x, e = self.propagate(edge_index,
                              query=None,
                              key=None,
                              value=None,
                              Xx=x,
                              Ee=e,
                              He=dist
                              )

        batch.x = self.activation(x) + x_in  # 更新节点特征

        if cfg.disableUpdateEdge and self.idx < cfg.limitedUpdateEdge:  # self.idx 表示只在前 x 个消息传递过程中更新边特征
            batch.edge_attr = e_in + e
        if not cfg.disableUpdateEdge:
            batch.edge_attr = e_in + e

        return batch

    """
    MessagePassing 基类会根据 propagate() 传入的参数，自动提取出源节点 Xx_i、目标节点 Xx_j
    """
    def message(self, Xx_i=None, Xx_j=None, Ee=None, He=None):  # 返回经过 Env 函数计算的边的权重
        """
        x_i           : [n_edges, dim_in]
        x_j           : [n_edges, dim_in]
        e             : [n_edges, dim_in]
        """
        e_ij = self.MLP_gate(torch.cat([Xx_i, Xx_j, Ee], dim=-1))

        if cfg.NormLayer == 'BN':
            e_ij = self.sigmoid(self.norm(e_ij))  # BN + Sigmoid
            # e_ij = F.sigmoid(self.norm(e_ij))  # BN + Sigmoid
        elif cfg.NormLayer == 'LN':
            e_ij = self.gelu(self.norm(e_ij))
        else:
            raise ValueError('NormLayer must be BN or LN')

        if self.use_envelope:
            self.alpha = self.envelope(He).unsqueeze(-1)
            sigma_ij = self.alpha * e_ij
        else:
            sigma_ij = e_ij

        self.e = sigma_ij

        return sigma_ij

    def aggregate(self, sigma_ij, index, Xx_i, Xx_j, Ee, Xx):  # 节点特征的聚集，加入 message 生成的消息权重
        """
        参数：
            sigma_ij：消息权重，形状为 [n_edges, dim_in]，来自 message 方法。
            index：中心节点索引，形状为 [n_edges]，即 edge_index[1]。
            Xx_i：中心节点特征，形状为 [n_edges, dim_in]。
            Xx_j：邻居节点特征，形状为 [n_edges, dim_in]。
            Ee：边特征，形状为 [n_edges, dim_in]。
            Xx：所有节点特征，形状为 [n_nodes, dim_in]，用于确定节点总数。
        说明：
            1. 形状为 [n_edges, dim_in] 的张量 sender
        """
        dim_size = Xx.shape[0]

        if cfg.onlyAtomFeaUpdate:
            sender = self.MLP_aggr(torch.cat([Xx_i, Xx_j], dim=-1))
        else:
            sender = self.MLP_aggr(torch.cat([Xx_i, Xx_j, Ee], dim=-1))

        out = scatter(sigma_ij*sender, index, 0, None, dim_size, reduce='sum')

        return out

    def update(self, aggr_out):
        """
        aggr_out        : [n_nodes, dim_in] ; is the output from aggregate() function after the aggregation
        x             : [n_nodes, dim_in]
        """
        x = self.norm2(aggr_out)
       
        e_out = self.e
        del self.e

        return x, e_out


class Readout(torch.nn.Module):
    def __init__(self,
        dim_in
    ):
        super(Readout, self).__init__()

        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in//2), 
                                nn.SiLU(inplace=True),
                                pyg_nn.Linear(dim_in//2, 1))

    def forward(self, batch):
        dim_size = int(batch.batch.max().item() + 1)  # 原子个数
        batch.x = self.MLP(batch.x)

        batch.x = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size).squeeze(-1)

        """ 先用平均池化，再用 MLP 
        batch.x = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size)
        batch.x = self.MLP(batch.x).squeeze(-1)  # 移除张量最后一个维度（如果它是 1）
        """
        return batch.x, batch.y

