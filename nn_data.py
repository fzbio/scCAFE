import gc
import glob
from inspect import isfunction
import numpy as np
from torch_geometric.data import Dataset
from os import path
import cooler
from schickit.data_reading import read_cool_as_sparse, read_loops_as_sparse, \
    read_multiple_cell_loops, read_multiple_cell_tad_scores
from schickit.data_storage import copy_coolers_from_scool
import pandas as pd
from torch_geometric.data import Data
from tqdm.auto import tqdm
from torch_geometric.utils import negative_sampling, remove_self_loops
from torch_geometric.transforms import BaseTransform
import torch
import copy
import os
from torchvision import transforms
import random
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from sklearn.preprocessing import StandardScaler
import joblib


def filter_range(edge_index, lower=10, higher=100):
    mask1 = torch.logical_and(edge_index[0, :] - edge_index[1, :] >= lower,
                              edge_index[0, :] - edge_index[1, :] <= higher)
    mask2 = torch.logical_and(edge_index[1, :] - edge_index[0, :] >= lower,
                              edge_index[1, :] - edge_index[0, :] <= higher)
    mask = torch.logical_or(mask1, mask2)
    edge_index = edge_index[:, mask]
    return edge_index



def easy_to_device(data, device, attrs_to_remove):
    if callable(data.keys):
        attrs = data.keys()
    else:
        attrs = data.keys
    for attr in attrs_to_remove:
        attrs.remove(attr)
    return data.to(device, *attrs, non_blocking=True)


class GaussianNoise(BaseTransform):
    def __init__(self, mean=0.0, std=0.2):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.x = data.x + torch.randn_like(data.x) * self.std + self.mean
        return data


class PositionalEncoding(BaseTransform):
    def __init__(self, dim=322):
        self.dim = dim
        self.p_enc_1d_model = Summer(PositionalEncoding1D(dim))

    def __call__(self, data):
        assert data.x is not None
        assert data.x.size()[1] == self.dim
        x = self.p_enc_1d_model(data.x[None, :, :])
        x = torch.squeeze(x, 0)
        data.x = x
        return data


class ReadKmerFeatures(BaseTransform):
    def __init__(self, kmer_input_path, chroms, is_train_set, scaler_path):
        self.kmer_path = kmer_input_path
        self.kmer_df = pd.read_csv(self.kmer_path, sep='\t', header=0, index_col=False)
        self.kmer_df = self.kmer_df[self.kmer_df['chrom'].isin(chroms)]
        feature_mat = self.kmer_df.iloc[:, 3:].to_numpy()
        if is_train_set:
            self.scaler = StandardScaler()
            self.scaler.fit(feature_mat)
            joblib.dump(self.scaler, scaler_path)
        else:
            self.scaler = joblib.load(scaler_path)
        feature_mat = self.scaler.transform(feature_mat)
        self.kmer_df.iloc[:, 3:] = feature_mat
        self.desired_chrom_dfs = {}
        for chrom in chroms:
            chrom_df = self.kmer_df[self.kmer_df['chrom'] == chrom]
            self.desired_chrom_dfs[chrom] = chrom_df

    def __call__(self, data):
        current_df = self.desired_chrom_dfs[data.chrom_name]
        assert len(current_df) == data.num_nodes
        mat = torch.FloatTensor(current_df.iloc[:, 3:].to_numpy())
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, mat], dim=-1)
        else:
            data.x = mat
        return data


class ReadMotifFeatures(BaseTransform):
    def __init__(self, motif_input_path, chroms, is_train_set, scaler_path):
        self.motif_path = motif_input_path
        self.motif_df = pd.read_csv(self.motif_path, sep='\t', header=0, index_col=False)
        self.motif_df = self.motif_df[self.motif_df['chrom'].isin(chroms)]
        feature_mat = self.motif_df.iloc[:, 3:].to_numpy()
        if is_train_set:
            self.scaler = StandardScaler()
            self.scaler.fit(feature_mat)
            joblib.dump(self.scaler, scaler_path)
        else:
            self.scaler = joblib.load(scaler_path)
        feature_mat = self.scaler.transform(feature_mat)
        self.motif_df.iloc[:, 3:] = feature_mat
        self.desired_chrom_dfs = {}
        for chrom in chroms:
            chrom_df = self.motif_df[self.motif_df['chrom'] == chrom]
            self.desired_chrom_dfs[chrom] = chrom_df

    def __call__(self, data):
        current_df = self.desired_chrom_dfs[data.chrom_name]
        assert len(current_df) == data.num_nodes
        mat = torch.FloatTensor(current_df.iloc[:, 3:].to_numpy())
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, mat], dim=-1)
        else:
            data.x = mat
        return data


class RemoveSelfLooping(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


def colwise_in(edge_index, loop_index, num_nodes):
    edge_index = edge_index.to('cpu')
    loop_index = loop_index.to('cpu')
    edge_index_vec = edge_index[0, :] * num_nodes + edge_index[1, :]
    loop_index_vec = loop_index[0, :] * num_nodes + loop_index[1, :]
    return (edge_index_vec[..., None] == loop_index_vec).any(-1)


def short_dist_neg_sampling(loop_index, contact_index, num_nodes, lower=10, higher=100, fix_seed=None):
    # Generate the same sampling if fix_seed is not None
    if fix_seed is not None:
        np.random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        random.seed(fix_seed)

    # This function is only for loop calling.
    original_device = loop_index.device
    contact_index = contact_index.to('cpu')
    loop_index = loop_index.to('cpu')
    contact_negs = contact_index[:, torch.logical_not(colwise_in(contact_index, loop_index, num_nodes))]

    # Limit the contact negs to lower higher range (consider both sides)
    mask1 = torch.logical_and(contact_negs[0, :] - contact_negs[1, :] >= lower,
                                contact_negs[0, :] - contact_negs[1, :] <= higher)
    mask2 = torch.logical_and(contact_negs[1, :] - contact_negs[0, :] >= lower,
                                contact_negs[1, :] - contact_negs[0, :] <= higher)
    mask = torch.logical_or(mask1, mask2)
    contact_negs = contact_negs[:, mask]

    if contact_negs.size(1) >= loop_index.size(1):
        random_indices = torch.randint(contact_negs.size(1), (loop_index.size(1),))
        contact_negs = contact_negs[:, random_indices]
        negs = contact_negs
    else:
        k_count = (higher - lower + 1) * 2
        short_neg_num = loop_index.size(1)
        per_k_short_neg_num = int(np.ceil(short_neg_num / k_count))
        short_neg_list = []
        for k in range(lower, higher + 1):
            i_range = (0, num_nodes - k)
            rand_rows = torch.randint(low=i_range[0], high=i_range[1], size=(1, per_k_short_neg_num))
            cols = rand_rows + k
            rand_index = torch.cat([rand_rows, cols], dim=0)
            short_neg_list.append(rand_index)
        for k in range(-higher, -lower + 1):
            i_range = (k, num_nodes)
            rand_rows = torch.randint(low=i_range[0], high=i_range[1], size=(1, per_k_short_neg_num))
            cols = rand_rows + k
            rand_index = torch.cat([rand_rows, cols], dim=0)
            short_neg_list.append(rand_index)
        short_neg = torch.cat(short_neg_list, dim=1)
        short_neg = short_neg[:, torch.logical_not(colwise_in(short_neg, torch.cat([loop_index, contact_negs], dim=1), num_nodes))]
        negs = torch.cat([short_neg, contact_negs], dim=1)
        if negs.size(1) >= loop_index.size(1):
            random_indices = torch.randint(negs.size(1), (loop_index.size(1),))
            negs = negs[:, random_indices]
    return negs.to(original_device)


# def short_dist_neg_sampling(loop_index, contact_index, num_nodes, lower=10, higher=100, fix_seed=None):
#     # Generate the same sampling if fix_seed is not None
#     if fix_seed is not None:
#         np.random.seed(fix_seed)
#         torch.manual_seed(fix_seed)
#         random.seed(fix_seed)
#
#     # This function is only for loop calling.
#     original_device = loop_index.device
#     contact_index = contact_index.to('cpu')
#     loop_index = loop_index.to('cpu')
#     contact_negs = contact_index[:, torch.logical_not(colwise_in(contact_index, loop_index, num_nodes))]
#     # Limit the contact negs to lower higher range (consider both sides)
#     mask1 = torch.logical_and(contact_negs[0, :] - contact_negs[1, :] >= lower,
#                                 contact_negs[0, :] - contact_negs[1, :] <= higher)
#     mask2 = torch.logical_and(contact_negs[1, :] - contact_negs[0, :] >= lower,
#                                 contact_negs[1, :] - contact_negs[0, :] <= higher)
#     mask = torch.logical_or(mask1, mask2)
#     contact_negs = contact_negs[:, mask]
#
#     k_count = (higher - lower + 1) * 2
#     short_neg_num = loop_index.size(1)
#     per_k_short_neg_num = 3 * int(np.ceil(short_neg_num / k_count))   # * 3 to make sure there are enough negative samples
#     short_neg_list = []
#     for k in range(lower, higher + 1):
#         i_range = (0, num_nodes - k)
#         rand_rows = torch.randint(low=i_range[0], high=i_range[1], size=(1, per_k_short_neg_num))
#         cols = rand_rows + k
#         rand_index = torch.cat([rand_rows, cols], dim=0)
#         short_neg_list.append(rand_index)
#     for k in range(-higher, -lower + 1):
#         i_range = (k, num_nodes)
#         rand_rows = torch.randint(low=i_range[0], high=i_range[1], size=(1, per_k_short_neg_num))
#         cols = rand_rows + k
#         rand_index = torch.cat([rand_rows, cols], dim=0)
#         short_neg_list.append(rand_index)
#     short_neg = torch.cat(short_neg_list, dim=1)
#     short_neg = short_neg[:, torch.logical_not(colwise_in(short_neg, torch.cat([loop_index, contact_negs], dim=1), num_nodes))]
#
#     if contact_negs.size(1) > loop_index.size(1):
#         random_indices = torch.randint(contact_negs.size(1), (loop_index.size(1),))
#         contact_negs = contact_negs[:, random_indices]
#
#     if short_neg.size(1) > loop_index.size(1):
#         random_indices = torch.randint(short_neg.size(1), (loop_index.size(1),))
#         short_neg = short_neg[:, random_indices]
#
#     negs = torch.cat([short_neg, contact_negs], dim=1)
#
#     # Finally, only preserve the length of loop_index.size(1)
#     if negs.size(1) >= loop_index.size(1):
#         random_indices = torch.randint(negs.size(1), (loop_index.size(1),))
#         negs = negs[:, random_indices]
#
#     return negs.to(original_device)


class ShortDistanceNegSampler(BaseTransform):
    # This class is only for loop calling.
    def __init__(self, ratio=1.0, fix_seed=None):
        self.ratio = ratio
        self.fix_seed = fix_seed

    def __call__(self, data):
        data.neg_edge_index = short_dist_neg_sampling(
            data.edge_label_index, data.edge_index, data.num_nodes, fix_seed=self.fix_seed
        )
        return data


def kth_diag_indices(num_nodes, k):
    rows, cols = np.diag_indices(num_nodes)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


# class NegativeSampling(BaseTransform):
#     def __init__(self, ratio=1.0, edge_type='contact'):
#         self.ratio = ratio
#         self.edge_type = edge_type
#
#     def __call__(self, data):
#         if self.edge_type == 'contact':
#             data.neg_edge_index = negative_sampling(
#                 data.edge_index, data.num_nodes,
#                 num_neg_samples=int(self.ratio * data.edge_index.size(1))
#             )
#         elif self.edge_type == 'loop':
#             data.neg_edge_index = negative_sampling(
#                 data.edge_label_index, data.num_nodes,
#                 num_neg_samples=int(self.ratio * data.edge_label_index.size(1))
#             )
#         else:
#             raise NotImplementedError('edge_type must be one of ["contact", "loop"].')
#         return data
#
#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}({self.ratio})'


def check_cell_types_same(cell_types1, cell_types2):
    if len(cell_types1) != len(cell_types2):
        return False
    cell_types1 = sorted(cell_types1)
    cell_types2 = sorted(cell_types2)
    for ct1, ct2 in zip(cell_types1, cell_types2):
        if ct1 != ct2:
            return False
    return True


class StreamScoolDataset(Dataset):
    """
    Streaming version of ScoolDataset
    """
    def __init__(self, root, scool_path,
                 chrom_names, resolution, celltype_bedpe_dict, celltype_tad_dict, name_parser=None,
                 desired_celltypes=None, weighted=True, transform=None, pre_transform=None, pre_filter=None):
        self.scool_path = scool_path
        self.chrom_names = chrom_names
        self.resolution = resolution
        self.name_parser = name_parser
        self.desired_celltpes = desired_celltypes
        self.weighted = weighted
        self.celltype_bedpe_dict = celltype_bedpe_dict
        self.celltype_tad_dict = celltype_tad_dict
        self.cell_names = cooler.fileops.list_scool_cells(self.scool_path)
        self.bins_selector = cooler.Cooler(self.scool_path + "::" + self.cell_names[0]).bins()
        chrom_df = cooler.Cooler(self.scool_path + "::" + self.cell_names[0]).chroms()[:]
        chrom_df = chrom_df[chrom_df['name'].isin(self.chrom_names)]
        self.chrom_df = pd.DataFrame({'name': self.chrom_names}).merge(chrom_df, on='name')
        assert check_cell_types_same(list(self.celltype_bedpe_dict), list(self.celltype_tad_dict))
        if name_parser is None:
            assert len(self.celltype_bedpe_dict) == 1
            self.cell_types = list(self.celltype_bedpe_dict) * len(self.cell_names)
        else:
            assert desired_celltypes is not None and callable(name_parser)
            all_cn, self.cell_names, self.cell_types = self.cell_names, [], []
            for c in all_cn:
                ct = name_parser(c, desired_celltypes)
                if ct is not None:
                    self.cell_types.append(ct)
                    self.cell_names.append(c)
        super().__init__(root, transform, pre_transform, pre_filter)
        self._loop_label_list = read_multiple_cell_loops(
            self.cell_types, copy.deepcopy(self.celltype_bedpe_dict), self.resolution, self.chrom_df, sparse_format='coo'
        )
        self._tad_label_list = read_multiple_cell_tad_scores(
            self.cell_types, copy.deepcopy(self.celltype_tad_dict), self.resolution, self.chrom_df
        )

    @property
    def processed_file_names(self):
        return []

    def process(self):
        pass

    def _process_item(self, idx):
        cell_i = idx // len(self.chrom_names)
        chrom_j = idx % len(self.chrom_names)
        cell_name = self.cell_names[cell_i]
        chrom_name = self.chrom_names[chrom_j]
        mat = cooler.Cooler(self.scool_path + '::' + cell_name).matrix(balance=False, sparse=True).fetch(chrom_name)
        loop = self._loop_label_list[cell_i * len(self.chrom_names) + chrom_j]
        tad_score = self._tad_label_list[cell_i * len(self.chrom_names) + chrom_j]
        graph = Data(x=None, num_nodes=mat.shape[0], edge_index=torch.tensor([mat.row, mat.col], dtype=torch.long))
        graph.edge_weights = torch.tensor(mat.data, dtype=torch.float) if self.weighted else None
        graph.edge_label_index = torch.tensor([loop.row, loop.col], dtype=torch.long)
        graph.tad_label = torch.tensor(tad_score, dtype=torch.float)
        graph.cell_name = cell_name
        graph.chrom_name = self.chrom_names[chrom_j]
        graph.cell_type = self.cell_types[cell_i] if self.cell_types is not None else None
        if self.pre_transform is not None:
            graph = self.pre_transform(graph)
        return graph

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return []

    def len(self):
        return len(self.cell_names) * len(self.chrom_names)

    def get(self, idx):
        data = self._process_item(idx)
        return data


def expand_cell_ids_to_graph_ids(indices, chrom_num):
    id_list = [indices + i for i in range(chrom_num)]
    id_vec = np.column_stack(id_list).flatten()
    # print(id_vec)
    return id_vec


def check_vec_unique(A):
    if np.unique(A).size < A.shape[0]:
        return False
    else:
        return True


def get_hmm_split_dataset(whole_dataset, hmm_train_ratio, chrom_list, seed=None):
    """
    Get train, val, and test sets for cross validation.
    The test set is one of the folds, and the validation set is selected randomly.
    This function is suitable for outer CV.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    N = len(whole_dataset)  # Len of dataset
    C = len(chrom_list)  # Len of chromosome list
    assert N % C == 0
    cell_num = N // C
    indices = np.arange(cell_num)
    indices = np.random.permutation(indices) * C

    train_indices = indices[:int(cell_num * hmm_train_ratio)]
    left_indices = indices[int(cell_num * hmm_train_ratio):]

    train_indices = expand_cell_ids_to_graph_ids(train_indices, C)
    left_indices = expand_cell_ids_to_graph_ids(left_indices, C)
    assert len(train_indices) + len(left_indices) == N
    assert check_vec_unique(np.concatenate([train_indices, left_indices]))
    trainset = whole_dataset.index_select(train_indices)
    left_set = whole_dataset.index_select(left_indices)
    return trainset, left_set


def get_cv_split_dataset(whole_dataset, n_fold, fold, val_set_ratio, chrom_list, seed):
    """
    Get train, val, and test sets for cross validation.
    The test set is one of the folds, and the validation set is selected randomly.
    This function is suitable for outer CV.
    """
    np.random.seed(seed)
    random.seed(seed)
    assert fold < n_fold
    N = len(whole_dataset)  # Len of dataset
    C = len(chrom_list)  # Len of chromosome list
    assert N % C == 0
    cell_num = N // C
    indices = np.arange(cell_num)
    indices = np.random.permutation(indices) * C

    bins = np.array(np.linspace(0, cell_num, n_fold + 1), dtype='int')
    test_indices = indices[bins[fold]:bins[fold + 1]]
    left_indices = np.random.permutation(np.concatenate([indices[:bins[fold]], indices[bins[fold + 1]:]]))
    train_size = int(len(left_indices) * (1 - val_set_ratio))
    train_indices = left_indices[:train_size]
    val_indices = left_indices[train_size:]
    train_indices = expand_cell_ids_to_graph_ids(train_indices, C)
    val_indices = expand_cell_ids_to_graph_ids(val_indices, C)
    test_indices = expand_cell_ids_to_graph_ids(test_indices, C)
    assert len(train_indices) + len(val_indices) + len(test_indices) == N
    assert check_vec_unique(np.concatenate([train_indices, val_indices, test_indices]))
    trainset = whole_dataset.index_select(train_indices)
    val_set = whole_dataset.index_select(val_indices)
    testset = whole_dataset.index_select(test_indices)
    return trainset, val_set, testset


def get_cv_split_scools(scool_path, n_fold, fold, val_set_ratio, seed,
                        train_scool_path, val_scool_path, test_scool_path):
    np.random.seed(seed)
    random.seed(seed)
    assert fold < n_fold
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    cell_num = len(cell_names)
    indices = np.arange(cell_num)
    indices = np.random.permutation(indices)

    bins = np.array(np.linspace(0, cell_num, n_fold + 1), dtype='int')
    test_indices = indices[bins[fold]:bins[fold + 1]]
    left_indices = np.random.permutation(np.concatenate([indices[:bins[fold]], indices[bins[fold + 1]:]]))
    train_size = int(len(left_indices) * (1 - val_set_ratio))
    train_indices = left_indices[:train_size]
    val_indices = left_indices[train_size:]

    assert len(train_indices) + len(val_indices) + len(test_indices) == cell_num
    assert check_vec_unique(np.concatenate([train_indices, val_indices, test_indices]))

    train_names = [cell_names[i] for i in train_indices]
    val_names = [cell_names[i] for i in val_indices]
    test_names = [cell_names[i] for i in test_indices]
    copy_coolers_from_scool(scool_path, train_scool_path, train_names)
    copy_coolers_from_scool(scool_path, val_scool_path, val_names)
    copy_coolers_from_scool(scool_path, test_scool_path, test_names)


def get_train_val_scools(scool_path, val_set_ratio, seed, train_scool_path, val_scool_path):
    np.random.seed(seed)
    random.seed(seed)
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    cell_num = len(cell_names)
    indices = np.arange(cell_num)
    indices = np.random.permutation(indices)

    train_size = int(len(indices) * (1 - val_set_ratio))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    assert len(train_indices) + len(val_indices) == cell_num
    assert check_vec_unique(np.concatenate([train_indices, val_indices]))

    train_names = [cell_names[i] for i in train_indices]
    val_names = [cell_names[i] for i in val_indices]
    copy_coolers_from_scool(scool_path, train_scool_path, train_names)
    copy_coolers_from_scool(scool_path, val_scool_path, val_names)


# def get_inner_cv_split_dataset(whole_dataset, n_fold, fold, test_set_ratio, chrom_list, seed):
#     """
#     Get train, val, and test sets for cross validation.
#     The test set is fixed for all folds. Select a slice for validation in each fold.
#     This function is suitable for inner CV.
#     """
#     np.random.seed(seed)
#     random.seed(seed)
#     assert fold < n_fold
#     N = len(whole_dataset)  # Len of dataset
#     C = len(chrom_list)  # Len of chromosome list
#     assert N % C == 0
#     cell_num = N // C
#     indices = np.arange(cell_num)
#     indices = np.random.permutation(indices) * C
#
#     test_size = int(cell_num * test_set_ratio)
#     test_indices = indices[cell_num - test_size:]
#
#     left_indices = indices[:cell_num - test_size]
#     bins = np.array(np.linspace(0, len(left_indices), n_fold + 1), dtype='int')
#     val_indices = left_indices[bins[fold]:bins[fold + 1]]
#     train_indices = np.concatenate([left_indices[:bins[fold]], left_indices[bins[fold + 1]:]])
#
#     train_indices = expand_cell_ids_to_graph_ids(train_indices, C)
#     val_indices = expand_cell_ids_to_graph_ids(val_indices, C)
#     test_indices = expand_cell_ids_to_graph_ids(test_indices, C)
#     assert len(train_indices) + len(val_indices) + len(test_indices) == N
#     assert check_vec_unique(np.concatenate([train_indices, val_indices, test_indices]))
#     trainset = whole_dataset.index_select(train_indices)
#     val_set = whole_dataset.index_select(val_indices)
#     test_set = whole_dataset.index_select(test_indices)
#     return trainset, val_set, test_set
#
#
# def get_inner_cv_split_scools(scool_path, n_fold, fold, test_set_ratio, seed,
#                         train_scool_path, val_scool_path, test_scool_path):
#     np.random.seed(seed)
#     random.seed(seed)
#     assert fold < n_fold
#     cell_names = cooler.fileops.list_scool_cells(scool_path)
#     cell_num = len(cell_names)
#     indices = np.arange(cell_num)
#     indices = np.random.permutation(indices)
#     test_size = int(cell_num * test_set_ratio)
#     test_indices = indices[cell_num - test_size:]
#     left_indices = indices[:cell_num - test_size]
#     bins = np.array(np.linspace(0, len(left_indices), n_fold + 1), dtype='int')
#     val_indices = left_indices[bins[fold]:bins[fold + 1]]
#     train_indices = np.concatenate([left_indices[:bins[fold]], left_indices[bins[fold + 1]:]])
#
#     assert len(train_indices) + len(val_indices) + len(test_indices) == cell_num
#     assert check_vec_unique(np.concatenate([train_indices, val_indices, test_indices]))
#
#     train_names = [cell_names[i] for i in train_indices]
#     val_names = [cell_names[i] for i in val_indices]
#     test_names = [cell_names[i] for i in test_indices]
#     copy_coolers_from_scool(scool_path, train_scool_path, train_names)
#     copy_coolers_from_scool(scool_path, val_scool_path, val_names)
#     copy_coolers_from_scool(scool_path, test_scool_path, test_names)



if __name__ == '__main__':
    print(check_cell_types_same({'HPC': 1, 'mES': 2}, {'HPC': 1, 'mES': 3}))
    print(check_cell_types_same({'HPC': 1, 'mES': 2}, {'HPC': 1, 'OPC': 2}))
    print(check_cell_types_same({'HPC': 1, 'mES': 2}, {'HPC': 1, 'mES': 2, 'OPC': 3}))
    print(check_cell_types_same({'mES': 2, 'HPC': 1}, {'HPC': 1, 'mES': 2}))
