import torch
from torch_geometric.graphgym.config import cfg


def create_model():
    if cfg.model == "AtomNet":
        from models.atomnet import AtomNet
        model = AtomNet(dim_in=cfg.dim_in,  # 256
                        dim_rbf=cfg.dim_rbf,  # 64
                        num_layers=cfg.num_layers,  # 4
                        invariant=cfg.invariant,  # False
                        use_envelope=cfg.envelope,  # True
                        atom_init=cfg.atom_init  # default: atom_number
                        ).to("cuda:0")

    else:
        raise Exception("Model not implemented")
    return model
