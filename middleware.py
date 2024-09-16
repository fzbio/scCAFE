from torch_geometric.data import Dataset, Data
from torch import nn
from nn_data import StreamScoolDataset
import torch
from multiscale_calling import MultitaskFeatureCaller
from configs import DEVICE
from configs import CompilationConfigs as cc
import tempfile


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