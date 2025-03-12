from torch_geometric.data import Dataset, Data
from torch import nn
from nn_data import StreamScoolDataset
import torch
from multiscale_calling import MultitaskFeatureCaller
from configs import DEVICE
from configs import CompilationConfigs as cc
import tempfile
import h5py
import torch


class CellDataset(Dataset):
    def __init__(self, h5file_path):
        tmp_dir = tempfile.TemporaryDirectory()
        super(CellDataset, self).__init__(root=tmp_dir.name)
        self.h5file_path = h5file_path
        with h5py.File(self.h5file_path, 'r') as f:
            self.cell_names = list(f.keys())
        tmp_dir.cleanup()

    @property
    def processed_file_names(self):
        return []

    def process(self):
        pass

    @torch.no_grad()
    def _process_item(self, idx):
        current_cell_name = self.cell_names[idx]
        with h5py.File(self.h5file_path, 'r') as f:
            x = f[current_cell_name]
            data = Data(
                x=torch.tensor(x, dtype=torch.float32).unsqueeze(0),
                cell_name=current_cell_name
            )
        return data

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return []

    def len(self):
        with h5py.File(self.h5file_path, 'r') as f:
            return len(f)

    def get(self, idx):
        data = self._process_item(idx)
        return data


class MiddleWareDataset(Dataset):
    """
    Streaming version of ScoolDataset
    """
    def __init__(self, scool_dataset: StreamScoolDataset, first_feature_caller: MultitaskFeatureCaller):
        tmp_dir = tempfile.TemporaryDirectory()
        super(MiddleWareDataset, self).__init__(root=tmp_dir.name)
        tmp_dir.cleanup()
        self.scool_dataset = scool_dataset
        self.first_feature_caller = first_feature_caller
        # Set requires_grad to False for all parameters in the model
        for param in self.first_feature_caller.vgae.parameters():
            param.requires_grad = False

    @property
    def processed_file_names(self):
        return []

    def process(self):
        pass

    @torch.no_grad()
    def _process_item(self, idx):
        self.first_feature_caller.vgae.eval()
        data = self.scool_dataset.get(idx)
        data = data.to(DEVICE)
        z = self.first_feature_caller.vgae.encode(data.x, data.edge_index)
        # z = torch.relu(self.first_feature_caller.vgae.encode(data.x, data.edge_index))

        middle_data = Data(x=z, tad_label=data.tad_label, chrom_name=data.chrom_name, cell_name=data.cell_name)
        return middle_data

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return []

    def len(self):
        return self.scool_dataset.len()

    def get(self, idx):
        data = self._process_item(idx)
        return data