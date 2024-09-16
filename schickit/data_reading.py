import pandas as pd
from scipy import sparse
import numpy as np
import cooler
import multiprocessing as mp
import itertools
from collections import defaultdict
from .utils import get_bin_count, grouplist
from .matrix_manipulation import weighted_hic_to_unweighted_graph
import random
from tqdm.auto import tqdm
import re


def check_chrom_in_order(chrom_df, chrom_names):
    chrom_df = chrom_df.reset_index(drop=True)
    indices = [chrom_df[chrom_df['name'] == c].index[0] for c in chrom_names]
    tmp = -1
    for i in indices:
        if i > tmp:
            tmp = i
        else:
            return False
    return True


def read_scool_as_sparse(scool_path, chrom_names, name_parser=None, desired_celltypes=None, weighted=True, workers=8, sparse_format='csc'):
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    if name_parser is None:
        cell_types = None
    else:
        assert desired_celltypes is not None and callable(name_parser)
        all_cn, cell_names, cell_types = cell_names, [], []
        for c in all_cn:
            ct = name_parser(c, desired_celltypes)
            if ct is not None:
                cell_types.append(ct)
                cell_names.append(c)

    with mp.Pool(workers) as pool:
        matrix_list = pool.starmap(
            read_cool_as_sparse,
            ((scool_path + "::" + cell_name, chrom_names, weighted, sparse_format) for cell_name in cell_names)
        )
    matrix_list = list(itertools.chain.from_iterable(matrix_list))

    # Assume all cells have the same bins dataframe
    bins_selector = cooler.Cooler(scool_path + "::" + cell_names[0]).bins()
    chrom_df = cooler.Cooler(scool_path + "::" + cell_names[0]).chroms()[:]
    assert check_chrom_in_order(chrom_df, chrom_names)
    chrom_df = chrom_df[chrom_df['name'].isin(chrom_names)]
    chrom_df = pd.DataFrame({'name': chrom_names}).merge(chrom_df, on='name')
    result_dict = {
        'cell_names': cell_names,
        'cell_types': cell_types,
        'chrom_df': chrom_df,
        'bins_selector': bins_selector,
        'matrix_list': matrix_list
    }
    if result_dict['cell_types'] is not None:
        assert len(result_dict['cell_types']) == len(result_dict['cell_names'])
    assert len(result_dict['matrix_list']) == len(result_dict['chrom_df']) * len(result_dict['cell_names'])
    return result_dict


def read_cool_multiple_chroms_as_sparse(scool_path, chrom_names, weighted=True):
    cooler_obj = cooler.Cooler(scool_path)
    pixels = cooler_obj.matrix(balance=False, as_pixels=True, join=True)[:]
    matrix_list = []
    for i, chrom_name in enumerate(chrom_names):
        # Select the pixels dataframe to only preserve the current chromosome.
        # columns chrom1 and chrom2 must be the same for the current chromosome.
        df = pixels[pixels['chrom1'] == chrom_name]
        df = df[df['chrom2'] == chrom_name]
        # Convert the pixels dataframe to a sparse matrix.
        resolution = cooler_obj.binsize
        num_bins = get_bin_count(cooler_obj.chromsizes[chrom_name], resolution)
        mat = sparse.coo_matrix(
            (df['count'], (df['start1'] // resolution, df['start2'] // resolution)),
            shape=(num_bins, num_bins)
        )
        mat = mat + sparse.triu(mat, k=1).transpose()
        if not weighted:
            mat = weighted_hic_to_unweighted_graph(mat)
        matrix_list.append(mat)
    return matrix_list


def read_cool_as_sparse(cool_path, chrom_names, weighted=True, sparse_format='csc'):
    cool_obj = cooler.Cooler(cool_path)
    matrix_list = []
    for chrom_name in chrom_names:
        mat = cool_obj.matrix(balance=False, sparse=True).fetch(chrom_name)
        if not weighted:
            mat = weighted_hic_to_unweighted_graph(mat)
        if sparse_format == 'coo':
            pass
        elif sparse_format == 'csc':
            mat = mat.tocsc()
        else:
            raise NotImplementedError('Unsupported sparse format.')
        matrix_list.append(mat)
    return matrix_list


def read_multiple_cell_tad_scores(cell_type_list, celltype_bed_dict, resolution, chrom_df):
    for ct in celltype_bed_dict:
        celltype_bed_dict[ct] = read_tad_score_from_bed(
            celltype_bed_dict[ct], chrom_df, resolution
        )
    vec_list = []
    for ct in cell_type_list:
        for i, chrom in enumerate(chrom_df['name']):
            chrom_len = chrom_df[chrom_df['name'] == chrom]['length'].iloc[0]
            assert get_bin_count(chrom_len, resolution) == len(celltype_bed_dict[ct][i])
        vec_list.append(celltype_bed_dict[ct])
    return list(itertools.chain.from_iterable(vec_list))


# def normalize_tad_score(tad_score):
#     standardized_score = (tad_score - np.mean(tad_score)) / np.std(tad_score)
#     return standardized_score


def read_tad_score_from_bed(bed_path, chrom_df, resolution):
    chrom_names = chrom_df['name']
    tad_score_df = pd.read_csv(
        bed_path, sep='\t', header=None, usecols=[0, 1, 2, 4], names=['chr', 'start', 'end', 'score']
    )
    tad_score_df['start'] = tad_score_df['start'] - 1

    score_vectors = []

    if len(tad_score_df) == 0:   # Deal with the placeholder case.
        for chrom_name in chrom_names:
            chrom_len = chrom_df[chrom_df['name'] == chrom_name]['length'].iloc[0]
            bin_count = get_bin_count(chrom_len, resolution)
            score_vectors.append(np.ones(bin_count, dtype='float') + 1)
    else:
        for chrom_name in chrom_names:
            df = tad_score_df[tad_score_df['chr'] == chrom_name]
            df = df.fillna(-1)
            pos_mask = df['score'] > 0
            neg_mask = df['score'] <= 0
            df.loc[pos_mask, 'score'] = 1
            df.loc[neg_mask, 'score'] = 0
            score_vector = df['score'].values
            # score_vector = normalize_tad_score(score_vector)
            score_vectors.append(score_vector)
    return score_vectors


def read_multiple_cell_loops(cell_type_list, celltype_bedpe_dict, resolution, chrom_df, sparse_format='csc'):
    for ct in celltype_bedpe_dict:
        celltype_bedpe_dict[ct] = read_loops_as_sparse(
            celltype_bedpe_dict[ct], resolution, chrom_df, sparse_format
        )
    matrix_list = []
    for ct in cell_type_list:
        matrix_list.append(celltype_bedpe_dict[ct])
    return list(itertools.chain.from_iterable(matrix_list))


def read_loops_as_sparse(bedpe_path, resolution, chrom_df, sparse_format='csc'):
    chrom_names = chrom_df['name']
    loop_dict = parsebed(bedpe_path, resolution)
    loop_dict = convert_loop_dict_to_symmetric(loop_dict)
    label_sp_matrix_list = []
    for chrom_name in chrom_names:
        if len(loop_dict) != 0:
            assert chrom_name in loop_dict
            chrom_bin_count = get_bin_count(chrom_df[chrom_df['name']==chrom_name]['length'].iloc[0], resolution)
            current_coord_list = loop_dict[chrom_name]
            indexes_1, indexes_2 = list(zip(*current_coord_list))
            current_coo = sparse.coo_matrix(
                ([1]*len(indexes_1), (indexes_1, indexes_2)),
                shape=(chrom_bin_count, chrom_bin_count)
            )
            label_sp_matrix_list.append(current_coo)
        else:
            chrom_bin_count = get_bin_count(chrom_df[chrom_df['name'] == chrom_name]['length'].iloc[0], resolution)
            current_coo = sparse.coo_matrix(
                ([], ([], [])), shape=(chrom_bin_count, chrom_bin_count)
            )
            label_sp_matrix_list.append(current_coo)
    if sparse_format == 'coo':
        pass
    elif sparse_format == 'csc':
        label_sp_matrix_list = [mat.tocsc() for mat in label_sp_matrix_list]
    else:
        raise NotImplementedError('Unsupported sparse format.')
    return label_sp_matrix_list


def convert_loop_dict_to_symmetric(loop_dict):
    symmetric_dict = dict()
    for chrom_name in loop_dict:
        symmetric_list = []
        for coord in loop_dict[chrom_name]:
            assert coord[0] < coord[1]
            a, b = coord[0], coord[1]
            symmetric_list.append((a, b))
            symmetric_list.append((b, a))
        assert len(symmetric_list) == 2 * len(loop_dict[chrom_name])
        symmetric_dict[chrom_name] = symmetric_list
    return symmetric_dict


def convert_loop_df_to_upper_triangle(df, deduplicate=True):
    df_upper = df.copy()
    df_upper = df_upper[df_upper['x1'] <= df_upper['y1']]
    df_lower = df.copy()
    df_lower = df_lower[df_lower['x1'] > df_lower['y1']]
    df_lower = df_lower.rename(columns={'x1': 'y1', 'x2': 'y2', 'y1': 'x1', 'y2': 'x2'})
    df_result = pd.concat([df_upper, df_lower], axis=0)
    if deduplicate:
        # Delete duplicated rows
        df_result = df_result.drop_duplicates()
    df_result = df_result.reset_index(drop=True)
    df_result = df_result[df.columns.tolist()]
    return df_result


def parsebed(chiafile, res=10000, lower=0, upper=5000000, valid_threshold=1):
    """
    Read the reference bedpe file and generate a distionary of positive center points.
    """
    coords = defaultdict(list)
    upper = upper // res
    with open(chiafile) as o:
        for line in o:
            s = line.rstrip().split()
            a, b = float(s[1]), float(s[4])
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            a = a // res
            b = b // res
            # all chromosomes including X and Y
            if (b - a > lower) and (b - a < upper) and 'M' not in s[0]:
                # always has prefix "chr", avoid potential bugs
                chrom = 'chr' + re.compile('^chr').sub('', s[0])
                coords[chrom].append((a, b))
    valid_coords = dict()
    for c in coords:
        current_set = set(coords[c])
        valid_set = set()
        for coord in current_set:
            if coords[c].count(coord) >= valid_threshold:
                valid_set.add(coord)
        valid_coords[c] = valid_set
    return valid_coords


def align_cell_dict_orders(reference_cell_names, cells_dict):
    assert len(reference_cell_names) == len(cells_dict['cell_names'])
    grp_matrix_list = grouplist(cells_dict['matrix_list'], len(cells_dict['chrom_df']))
    reference_df = pd.DataFrame({'cell_names': reference_cell_names})
    right_df = pd.DataFrame({'cell_names': cells_dict['cell_names']})
    right_df['new_id'] = right_df.index
    df = reference_df.merge(right_df, on='cell_names')
    order = df['new_id']
    matrix_list = list(itertools.chain.from_iterable([grp_matrix_list[i] for i in order]))
    cell_names = reference_cell_names.copy()
    cell_types = [cells_dict['cell_types'][i] for i in order]
    return {
        'cell_names': cell_names,
        'cell_types': cell_types,
        'chrom_df': cells_dict['chrom_df'],
        'bins_selector': cells_dict['bins_selector'],
        'matrix_list': matrix_list
    }


