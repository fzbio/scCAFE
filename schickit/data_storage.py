import cooler
import pandas as pd
import numpy as np
import pickle
import os
import multiprocessing as mp
from schickit.utils import grouplist
from scipy import sparse
from tqdm.auto import tqdm
import random
from .utils import CHUNK_SIZE
import re



def random_select_subset_scools(scool_path, output_path, cell_num, seed):
    random.seed(seed)
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    random.shuffle(cell_names)
    cell_names = cell_names[:cell_num]
    copy_coolers_from_scool(scool_path, output_path, cell_names)


def select_types_of_cells_from_scool(scool_path, output_path, name_parser):
    """
    name_parser is a function that determines if a cell name is of our desired cell type. It returns a boolean value.
    """
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    selected_cell_names = [name for name in cell_names if name_parser(name)]
    copy_coolers_from_scool(scool_path, output_path, selected_cell_names)


def copy_from_scool_to_mcool_dir(in_scool_path, out_mcool_dir, resolutions, cell_names=None, parser=lambda x: x.split('/')[-1]):
    if cell_names is None:
        cell_names = cooler.fileops.list_scool_cells(in_scool_path)
    for cell_name in tqdm(cell_names):
        mcool_file_name = parser(cell_name) + '.mcool'
        mcool_path = os.path.join(out_mcool_dir, mcool_file_name)
        cooler.zoomify_cooler(in_scool_path + '::' + cell_name, mcool_path, resolutions, CHUNK_SIZE)

def copy_coolers_from_mcool_list(in_mcool_paths_list, out_scool_path, resolution, parser=lambda x: re.compile(r'\.mcool$').sub('', x)):
    for mcool_path in tqdm(in_mcool_paths_list):
        clr = cooler.Cooler(mcool_path + '::/resolutions/{}'.format(resolution))
        bins = clr.bins()[:]
        pixels = clr.pixels()[:]
        cell_name = parser(os.path.basename(mcool_path))
        cooler.create_scool(
            out_scool_path, {cell_name: bins}, {cell_name: pixels}, mode='a', symmetric_upper=True,
            dupcheck=True, triucheck=True, ensure_sorted=True
        )


def copy_coolers_from_scool(in_scool_path, out_scool_path, cell_names, lazy=True):
    clr = cooler.Cooler(in_scool_path)
    bins = clr.bins()[:]
    if not lazy:
        for name in cell_names:
            pixels = cooler.Cooler(in_scool_path + '::' + name).pixels()[:]
            cooler.create_scool(
                out_scool_path, {name: bins}, {name: pixels}, mode='a', symmetric_upper=True,
                dupcheck=True, triucheck=True, ensure_sorted=True
            )
    else:
        # Create the scool using the first cell
        pixels = cooler.Cooler(in_scool_path + '::' + cell_names[0]).pixels()[:]
        cooler.create_scool(
            out_scool_path, {cell_names[0]: bins}, {cell_names[0]: pixels}, mode='w', symmetric_upper=True,
            dupcheck=True, triucheck=True, ensure_sorted=True
        )
        # Append the rest cells
        for name in cell_names[1:]:
            cooler.fileops.cp(in_scool_path + '::' + name, out_scool_path + '::' + name)


def save_to_pickle(output_dir, cell_name, matrix_dicts):
    pickle.dump(
        matrix_dicts,
        open(os.path.join(output_dir, "{}.pkl".format(cell_name)), "wb")
    )


def create_pixel_df(matrices_of_cell, bins_selector, chroms):
    sanitizer = cooler.create.sanitize_pixels(bins_selector, sort=True, tril_action='raise')
    dfs = []
    for i, m in enumerate(matrices_of_cell):
        chrom = chroms[i]
        bin_1 = np.asarray(bins_selector[bins_selector['chrom'] == chrom].index.values[m.row], dtype='int32')
        bin_2 = np.asarray(bins_selector[bins_selector['chrom'] == chrom].index.values[m.col], dtype='int32')
        df = pd.DataFrame({'bin1_id': bin_1, 'bin2_id': bin_2, 'count': m.data})
        dfs.append(df)
    return sanitizer(pd.concat(dfs).reset_index(drop=True))


def save_sparse_to_scool(cells_dict, chroms, out_scool, workers=10):
    """
    This function may cause memory overflow when the number of cells is very large.
    To avoid this, use add_sparse_to_scool instead.
    """
    cell_names = cells_dict['cell_names']
    bins = cells_dict['bins_selector'][:]
    matrix_list = cells_dict['matrix_list']
    matrix_list = [sparse.triu(m, k=0, format='coo') for m in matrix_list]
    matrix_list = grouplist(matrix_list, len(chroms))
    with mp.Pool(workers) as pool:
        pixel_dfs = pool.starmap(create_pixel_df, [(group, bins, chroms) for group in matrix_list])
    cell_name_pixels_dict = {cell_names[i]: pixel_dfs[i] for i in range(len(cell_names))}
    cooler.create_scool(out_scool, bins, cell_name_pixels_dict, ordered=True, symmetric_upper=True, mode='w', triucheck=True, dupcheck=True, ensure_sorted=True)


def add_sparse_to_scool(cells_dict, chroms, out_scool):
    cell_names = cells_dict['cell_names']
    bins = cells_dict['bins_selector'][:]
    matrix_list = cells_dict['matrix_list']
    matrix_list = [sparse.triu(m, k=0, format='coo') for m in matrix_list]
    matrix_list = grouplist(matrix_list, len(chroms))
    assert len(matrix_list) == len(cell_names)
    for i, group in enumerate(tqdm(matrix_list)):
        pixel_df = create_pixel_df(group, bins, chroms)
        cell_name_pixels_dict = {cell_names[i]: pixel_df}
        cooler.create_scool(out_scool, bins, cell_name_pixels_dict, ordered=True, symmetric_upper=True, mode='a', triucheck=True, dupcheck=True, ensure_sorted=True)


def save_sparse_to_cool(cool_path, chroms, matrix_list, bins_selector):
    matrix_list = [sparse.triu(m, k=0, format='coo') for m in matrix_list]
    bins = bins_selector[:]
    pixel_df = create_pixel_df(matrix_list, bins, chroms)
    cooler.create_cooler(cool_path, bins, pixel_df, ordered=True, symmetric_upper=True, mode='w', triucheck=True, ensure_sorted=True)


if __name__ == '__main__':
    copy_coolers_from_scool(
        '../data/mES/nagano_100kb.scool', '../data/mES/nagano_100kb.cp.scool',
        ['/cells/1CDS1_1', '/cells/1CDS1_2']
    )