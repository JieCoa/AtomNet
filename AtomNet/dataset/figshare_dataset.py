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
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

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

    ----------
    elem_embedding_file: str. The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)

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
        root: The root directory tells the storage location of the base class dataset. root = "./dataset/jarvis/"

        InMemoryDataset will set raw_dir (raw data directory) and processed_dir (processed data directory) based on root.
         - In the initialization of the InMemoryDataset, processed_paths are calculated through the processed_file_names attribute.
         - Default behavior:
            If the subclass does not override processed_file_names, InMemoryDataset assumes the default file name as "data.pt".
            Therefore, self.processed_paths[0] is usually rootprocesseddata.pt.
        """
        if atom_init != 'atom_number':
            self.json_file = atom_init + '.json'
            self.name = name + '_' + self.json_file  # "dft_3d_2021_5_25_formation_energy_peratom_123_train" + '_' + json_file
            atom_init_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json/', self.json_file)

            assert os.path.exists(atom_init_file), f'{self.json_file} does not exist!'
        else:
            self.name = name

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

        # If there is data with the same name (self.name) in jarvisprocessed, the following process method will not be executed.
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
        
        return data

    def process(self):
        data_list = []
        scaler = StandardScaler()
        for i, (ddat, target) in tqdm(enumerate(zip(self.data, self.targets)), total=len(self.data)):
            structure = Atoms.from_dict(ddat["atoms"])

            if hasattr(self, "json_file"):
                atomic_numbers = [get_node_attributes(s, atom_features="atomic_number") for s in structure.elements]
                atom_features = np.vstack([self.ari.get_atom_fea(i[0]) for i in atomic_numbers])
                atom_features = torch.tensor(atom_features)

                data = Data(x=atom_features, y=target)
            else:
                atomic_numbers = torch.tensor([get_node_attributes(s, atom_features="atomic_number") for s in structure.elements]).squeeze(-1)

                data = Data(x=atomic_numbers, y=target)

            # data.eneg = torch.tensor([element(i[0]).electronegativity(scale='sanderson') for i in atomic_numbers])
            data.pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
            data.cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

            data.pbc = torch.tensor([[True, True, True]])
            data.natoms = torch.tensor([data.x.shape[0]])

            data.cell = data.cell.unsqueeze(0)

            # Compute PBC
            batch = Batch.from_data_list([data])

            edge_index, _, _, cart_vector = radius_graph_pbc(batch, self.radius, self.max_neigh)
            
            data.cart_dist = torch.norm(cart_vector, p=2, dim=-1)
            data.cart_dir = torch.nn.functional.normalize(cart_vector, p=2, dim=-1)

            data.edge_index = edge_index
            if len(edge_index[0]) == 0:
                continue

            if cfg.useElectronegativity:
                diffs = []
                for i in range(edge_index.shape[1]):
                    e1 = self.eneg_dict[atomic_numbers[edge_index[0][i]][0] - 1]
                    e2 = self.eneg_dict[atomic_numbers[edge_index[1][i]][0] - 1]
                    diff = abs(e1 - e2)
                    diffs.append(diff)
                if cfg.normalizedElectronegativity and len(diffs) > 0:
                    diffs = scaler.fit_transform(np.array(diffs).reshape(-1, 1))
                    data.eneg = torch.tensor(diffs, dtype=torch.float32)
                else:
                    data.eneg = torch.tensor(diffs, dtype=torch.float32).unsqueeze(-1)

            delattr(data, "pbc")
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])