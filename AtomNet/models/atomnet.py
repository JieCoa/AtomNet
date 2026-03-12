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

        self.encoder = Encoder(dim_in, dim_rbf=dim_rbf, radius=radius, invariant=invariant, atom_init=atom_init)
        self.dim_in = dim_in  # 256

        layers = []
        for i in range(num_layers):  # for _ in range(num_layers):
            layers.append(AtomNet_layer(
                idx=i,  # Perform a finite number of edge feature updates based on the number of layers
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
        dim_in: int,
        dim_rbf: int,
        radius: float = 5.0,
        invariant: bool = False,
        atom_init: str = "atom_number",
    ):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.invariant = invariant
        self.atom_init = atom_init

        if re.search(r'\d+', atom_init):
            # number = int(re.search(r'\d+', atom_init).group())
            number = int(re.search(r'(\d+)d', atom_init).group(1))
            self.embedding = nn.Linear(number, self.dim_in * 2)

        elif atom_init == 'atom_number':
            self.embedding = nn.Embedding(119, self.dim_in * 2)  # 119 → 512

        elif atom_init == 'cgcnn':
            self.embedding = nn.Linear(92, self.dim_in * 2)  # 92 → 512

        else:
            assert "The corresponding node feature dimension was not identified!"

        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        self.bias = nn.Parameter(torch.zeros(self.dim_in*2))
        self.activation = nn.SiLU(inplace=True)

        self.encoder_atom = nn.Sequential(self.activation,
                                          pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                          self.activation,
                                          )
        if self.invariant:
            dim_edge = dim_rbf
        elif cfg.usePolynomial > 0:
            dim_edge = dim_rbf + 3 + cfg.usePolynomial
        else:
            dim_edge = dim_rbf + 3
        if cfg.useElectronegativity:
            if cfg.normalized_Polynomial or cfg.eneg_type is None:
                dim_edge += 1
            elif cfg.eneg_type == 'newRBF':
                dim_edge += 20
            elif cfg.eneg_type is not None:
                dim_edge += 64  # newRBF02 & newRBF03 & newRBF04 & newRBF05
            else:
                raise ValueError("The corresponding type of electronegativity difference was not identified!")


        self.encoder_edge = nn.Sequential(pyg_nn.Linear(dim_edge, self.dim_in*2),
                                        self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)

        self.rbf = ExpNormalSmearing(0.0, radius, dim_rbf, False)  # Initialize the RBF (radial basis function)

        if cfg.useElectronegativity and cfg.eneg_type is not None:  # newRBF & newRBF02 & newRBF03 & newRBF04 & newRBF05
            self.rbf_eneg = RBF2Electronegativity(cutoff_upper=2.14, type=cfg.eneg_type, trainable=False)

    def forward(self, batch):
        if self.atom_init == 'atom_number':
            x = self.embedding(batch.x) + self.bias
        else:
            x = self.embedding(batch.x.float()) + self.bias

        batch.x = self.encoder_atom(x)

        if cfg.invariant:
            batch.edge_attr = self.encoder_edge(self.rbf(batch.cart_dist))
        elif cfg.usePolynomial > 0:
            dist = batch.cart_dist.unsqueeze(-1)
            if cfg.normalized_Polynomial:
                dist_min = dist.min(dim=0, keepdim=True).values
                dist_max = dist.max(dim=0, keepdim=True).values
                dist_normalized = (dist - dist_min) / (dist_max - dist_min + 1e-8)
                polynomial = [dist_normalized ** i for i in range(1, cfg.usePolynomial + 1)]
            else:
                polynomial = [dist ** i for i in range(1, cfg.usePolynomial + 1)]

            polynomial = torch.cat(polynomial, dim=-1)

            if cfg.useElectronegativity:
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
        else:
            if cfg.useElectronegativity:
                if cfg.eneg_type is None:
                    batch.edge_attr = self.encoder_edge(
                        torch.cat([self.rbf(batch.cart_dist), batch.cart_dir, batch.eneg], dim=-1))
                else:
                    batch.edge_attr = self.encoder_edge(
                        torch.cat([self.rbf(batch.cart_dist), batch.cart_dir, self.rbf_eneg(batch.eneg)], dim=-1))
            else:
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
        self.edge_batch = None
        self.idx = idx
        self.dim_in = dim_in
        self.activation = nn.SiLU(inplace=True)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.1)
        # self.softmax = nn.Softmax(dim=-1)

        if cfg.onlyAtomFeaUpdate:
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
        edge_index      : [2, n_edges] == [[neighbor atom index], [central atom index]]
        dist            : [n_edges]
        batch           : [n_nodes]
        """
        x_in = x
        e_in = e

        x, e = self.propagate(edge_index, Xx=x, Ee=e, He=dist)

        batch.x = self.activation(x) + x_in

        if cfg.disableUpdateEdge and self.idx < cfg.limitedUpdateEdge:
            batch.edge_attr = e_in + e
        if not cfg.disableUpdateEdge:
            batch.edge_attr = e_in + e

        return batch

    def message(self, Xx_i=None, Xx_j=None, Ee=None, He=None):
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

    def aggregate(self, sigma_ij, index, Xx_i, Xx_j, Ee, Xx):
        """
        Parameter：
            sigma_ij：message weights, in the shape of [n_edges, dim_in], come from the message method.
            index：The central node index, in the shape of [n_edges], is edge_index[1].
            Xx_i：The central node feature is in the shape of [n_edges, dim_in].
            Xx_j：Neighbor node features, with the shape of [n_edges, dim_in].
            Ee：Edge features, with the shape of [n_edges, dim_in].
            Xx：All node features, in the shape of [n_nodes, dim_in], are used to determine the total number of nodes.
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
        dim_size = int(batch.batch.max().item() + 1)
        batch.x = self.MLP(batch.x)

        batch.x = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size).squeeze(-1)

        return batch.x, batch.y
