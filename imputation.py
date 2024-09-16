import os.path

import cooler
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from schickit.file_format_conversion import convert_cool_to_scool
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
import tempfile
from schickit.utils import CHUNK_SIZE
from schickit.data_storage import copy_from_scool_to_mcool_dir
import re

# from scHiCTools import scHiCs
from hack_scHiCTools.hacked_schictools import scHiCs

def impute_and_save_cell(cell_name, closest_cell_names, zoom_in_scool_path, tmp_dir):
    out_uri = os.path.join(tmp_dir, cell_name.split('/')[-1] + '.cool')
    input_uris = [zoom_in_scool_path + '::' + c for c in [cell_name] + closest_cell_names]
    cooler.merge_coolers(out_uri, input_uris, CHUNK_SIZE, agg={'count': 'mean'})


def convert_dist_mat_to_closest_cells(cell_names, dist_matrix, k):
    closest_cells_list = []
    for i in range(len(dist_matrix)):
        dist_vect = dist_matrix[:, i]
        top_k_indices = np.argsort(dist_vect)[:k]
        closest_cells_list.append([cell_names[top_i] for top_i in top_k_indices])
    return closest_cells_list


def get_cell_names_of_ds(ds):
    cell_names = []
    unique_names = []
    loader = DataLoader(ds, shuffle=False, batch_size=1)
    for d in loader:
        cell_names.append(d.cell_name[0])
    # print(cell_names)
    for n in cell_names:
        if n not in unique_names:
            unique_names.append(n)
    return unique_names


def check_mcool_file_paths_order_same_with_cell_names(cell_names, mcool_file_paths):
    for i in range(len(cell_names)):
        if cell_names[i].split('/')[-1] != re.compile(r'\.mcool$').sub('', mcool_file_paths[i].split('/')[-1]):
            return False
    return True

class Imputer(object):
    def __init__(self, k=3):
        self.k = k


    def impute_dataset(self, in_finer_scool, out_finer_scool, chroms, assembly, tmp_dir_root, parser=lambda x: x.split('/')[-1]):
        with tempfile.TemporaryDirectory(dir=tmp_dir_root) as mcool_dir:
            cell_names = cooler.fileops.list_scool_cells(in_finer_scool)
            print('Coarsening...')
            copy_from_scool_to_mcool_dir(in_finer_scool, mcool_dir, [100000], parser=parser)
            print('Done.')
            mcool_file_paths = [os.path.join(mcool_dir, parser(f) + '.mcool') for f in cell_names]
            # chrom_rename_dict = {chrom: chrom.lstrip('chr') for chrom in chroms}
            # for mcool_file in mcool_file_paths:
            #     cooler.rename_chroms(cooler.Cooler(mcool_file + '::/resolutions/100000'), chrom_rename_dict)
            if not check_mcool_file_paths_order_same_with_cell_names(cell_names, mcool_file_paths):
                print(cell_names)
                print(mcool_file_paths)
            assert check_mcool_file_paths_order_same_with_cell_names(cell_names, mcool_file_paths)

            x = scHiCs(
                mcool_file_paths,
                reference_genome=assembly,
                resolution=100000,
                max_distance=4000000,
                format='.mcool',
                adjust_resolution=True,
                chromosomes=chroms,
                operations=['convolution'],
                kernel_shape=3,
                keep_n_strata=10,
                store_full_map=False
            )
            emb, _ = x.learn_embedding(similarity_method='innerproduct',
                                   embedding_method='MDS',
                                   aggregation='median',
                                   print_time=False,
                                   return_distance=True)
            dist_matrix = cdist(emb, emb)

        np.fill_diagonal(dist_matrix, 99999)
        closest_cells_list = convert_dist_mat_to_closest_cells(cell_names, dist_matrix, self.k)
        assert len(closest_cells_list) == len(cell_names)
        with tempfile.TemporaryDirectory(dir=tmp_dir_root) as temp_dir:
            print('\nImputing...')
            for i in tqdm(list(range(len(cell_names)))):
                impute_and_save_cell(cell_names[i], closest_cells_list[i],
                                     in_finer_scool, temp_dir)
            print('Done')
            print('Creating .scool from imputed coolers...')
            convert_cool_to_scool(temp_dir, out_finer_scool, lambda x: re.compile(r'\.cool$').sub('', x.split('/')[-1]))

