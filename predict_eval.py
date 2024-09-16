#!/usr/bin/env python3
import glob
import time

import pandas as pd

import numpy as np
import random
from nn_data import StreamScoolDataset, expand_cell_ids_to_graph_ids, ReadKmerFeatures, ReadMotifFeatures, PositionalEncoding
from torch_geometric import transforms as T
from nn_data import RemoveSelfLooping
from metrics import slack_metrics_df, slack_f1_df
import os
from sklearn.preprocessing import minmax_scale
from multiscale_calling import CrossStitchFeatureCaller, MultitaskFeatureCaller
from inference_configs import SelectedHyperparameters as hyperparams
from inference_configs import DEVICE
from scipy.sparse import coo_matrix
from schickit.utils import get_chrom_sizes, get_bin_count
from skimage.feature import peak_local_max, corner_peaks
from tqdm.auto import tqdm
from nn_data import kth_diag_indices
from scipy.signal import convolve2d
import time
from matplotlib import pyplot as plt
from middleware import MiddleWareDataset
from tad_calling import TadCaller
from compartment import CompartmentCaller


def predict_compartment_on_other_dataset(trained_model_dir, run_id, chroms, bedpe_dict, tad_dict, finer_scool_path,
                                    graph_dataset_path, compartment_out_dir, kmer_feature_path,
                                    motif_feature_path, chrom_sizes_path, resolution, assembly_path,
                                    name_parser=None, desired_cell_type=None):
    graph_dataset = StreamScoolDataset(
        graph_dataset_path,
        finer_scool_path,
        chroms, 10000,
        bedpe_dict, tad_dict,
        name_parser, desired_cell_type,
        pre_transform=T.Compose([
            RemoveSelfLooping(),
            ReadKmerFeatures(
                kmer_feature_path, chroms, False,
                os.path.join(trained_model_dir, f'{run_id}_kmer_scaler_calling.pkl')
            ),
            ReadMotifFeatures(
                motif_feature_path, chroms, False,
                os.path.join(trained_model_dir, f'{run_id}_motif_scaler_calling.pkl')
            ),
            PositionalEncoding()
        ])
    )
    feature_caller = MultitaskFeatureCaller(
        run_id, chroms, f'{trained_model_dir}/{run_id}.pt', graph_dataset.num_features,
        hyperparams.alpha, hyperparams.beta
    )
    feature_caller.load_model()
    test_set = MiddleWareDataset(graph_dataset, feature_caller)
    compartment_caller = CompartmentCaller()
    compartment_caller.predict_compartments(
        test_set, assembly_path, chrom_sizes_path, resolution, compartment_out_dir)


def predict_tads_on_other_dataset(trained_model_dir, run_id, chroms, bedpe_dict, tad_dict, finer_scool_path,
                                    graph_dataset_path, tad_out_dir, kmer_feature_path,
                                    motif_feature_path, chrom_sizes_path, resolution,
                                    name_parser=None, desired_cell_type=None):
        graph_dataset = StreamScoolDataset(
            graph_dataset_path,
            finer_scool_path,
            chroms, 10000,
            bedpe_dict, tad_dict,
            name_parser, desired_cell_type,
            pre_transform=T.Compose([
                RemoveSelfLooping(),
                ReadKmerFeatures(
                    kmer_feature_path, chroms, False,
                    os.path.join(trained_model_dir, f'{run_id}_kmer_scaler_calling.pkl')
                ),
                ReadMotifFeatures(
                    motif_feature_path, chroms, False,
                    os.path.join(trained_model_dir, f'{run_id}_motif_scaler_calling.pkl')
                ),
                PositionalEncoding()
            ])
        )
        feature_caller = MultitaskFeatureCaller(
            run_id, chroms, f'{trained_model_dir}/{run_id}.pt', graph_dataset.num_features,
            hyperparams.alpha, hyperparams.beta
        )
        feature_caller.load_model()
        test_set = MiddleWareDataset(graph_dataset, feature_caller)

        tad_caller = TadCaller(motif_feature_path)
        tad_caller.predict(
            tad_out_dir, test_set, DEVICE, get_chrom_sizes(chrom_sizes_path), resolution
        )




def predict_on_other_dataset(trained_model_dir, run_id, chroms, bedpe_dict, tad_dict, finer_scool_path,
                             graph_dataset_path, thresh, loop_out_dir, tad_out_dir, kmer_feature_path,
                             motif_feature_path, chrom_sizes_path,
                             name_parser=None, desired_cell_type=None, output_embedding=None):
    graph_dataset = StreamScoolDataset(
        graph_dataset_path,
        finer_scool_path,
        chroms, 10000,
        bedpe_dict, tad_dict,
        name_parser, desired_cell_type,
        pre_transform=T.Compose([
            RemoveSelfLooping(),
            ReadKmerFeatures(
                kmer_feature_path, chroms, False,
                os.path.join(trained_model_dir, f'{run_id}_kmer_scaler_calling.pkl')
            ),
            ReadMotifFeatures(
                motif_feature_path, chroms, False,
                os.path.join(trained_model_dir, f'{run_id}_motif_scaler_calling.pkl')
            ),
            PositionalEncoding()
        ])
    )
    feature_caller = MultitaskFeatureCaller(
        run_id, chroms, f'{trained_model_dir}/{run_id}.pt', graph_dataset.num_features,
        hyperparams.alpha, hyperparams.beta
    )
    feature_caller.load_model()
    feature_caller.predict(
        loop_out_dir, graph_dataset, DEVICE, thresh, output_embedding=output_embedding
    )



def read_bedpe_as_df(bedpe_path):
    label_df = pd.read_csv(
        bedpe_path, header=None, index_col=False, sep='\t', dtype={0: 'str', 3: 'str'},
        names=['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2']
    )
    label_df['chrom1'], label_df['chrom2'] = 'chr' + label_df['chrom1'], 'chr' + label_df['chrom2']
    return label_df


def get_raw_average_pred_dfs(pred_dfs, chrom_sizes_path, total, res=10000, sc_loop_threshold=0.5):
    # This is a version using coo matrix of scipy
    chrom_matrices = {}
    chrom_sizes = get_chrom_sizes(chrom_sizes_path)
    for df in tqdm(pred_dfs, total=total):
        df = df[df['proba'] > sc_loop_threshold]
        cell_weight = np.sum(df['proba'])
        df.loc[:, ['x1', 'x2', 'y1', 'y2']] = df[['x1', 'x2', 'y1', 'y2']].astype('int')
        for chrom in df['chrom1'].unique():
            mat_shape = get_bin_count(chrom_sizes[chrom], res)
            # Convert the df to a coo matrix
            chrom_df = df[df['chrom1'] == chrom]
            chrom_df = chrom_df[['x1', 'x2', 'y1', 'y2', 'proba']]
            row, col = chrom_df['x1'].to_numpy() // res, chrom_df['y1'].to_numpy() // res
            data = chrom_df['proba'].to_numpy() * cell_weight
            mat = coo_matrix((data, (row, col)), shape=(mat_shape, mat_shape))
            if chrom not in chrom_matrices:
                chrom_matrices[chrom] = mat
            else:
                chrom_matrices[chrom] += mat
    # Convert the coo matrix back to a df
    result_dfs = []
    for chrom in chrom_matrices:
        mat = chrom_matrices[chrom].tocoo()
        row, col, data = mat.row, mat.col, mat.data
        df = pd.DataFrame({'x1': row * res, 'x2': row * res + res, 'y1': col * res, 'y2': col * res + res, 'proba': data})
        df['chrom1'], df['chrom2'] = chrom, chrom
        df = df[['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2', 'proba']]
        result_dfs.append(df)
    return pd.concat(result_dfs)


def evaluate_average_cells(cell_pred_paths, bedpe_path, resolution, chrom_sizes_path, loop_num=None, threshold=None, percentile=None, sc_loop_threshold=0.5):
    """
    Evaluate based on the average prediction of a cell type
    cell_pred_paths must be of the same cell type
    """
    label_df = read_bedpe_as_df(bedpe_path)
    average_pred = get_average_preds(cell_pred_paths, bedpe_path, resolution, chrom_sizes_path, loop_num, threshold, percentile, sc_loop_threshold)
    candidate_df = average_pred.drop('proba', axis=1)
    # print(len(candidate_df))
    return slack_metrics_df(label_df, candidate_df, resolution) + (len(candidate_df),) + (average_pred,)


def get_average_preds(cell_pred_paths, bedpe_path, resolution, chrom_sizes_path, loop_num=None, threshold=None, percentile=None, sc_loop_threshold=0.5):
    """
    Evaluate based on the average prediction of a cell type
    cell_pred_paths must be of the same cell type
    """
    pred_dfs = []
    for pred_path in cell_pred_paths:
        pred_df = pd.read_csv(pred_path, header=0, index_col=False, sep='\t')
        pred_df = pred_df[pred_df['proba'] >= sc_loop_threshold].reset_index(drop=True)
        pred_dfs.append(pred_df)
    time_start = time.time()
    average_pred = get_raw_average_pred_dfs(pred_dfs, chrom_sizes_path, len(cell_pred_paths), resolution, sc_loop_threshold=sc_loop_threshold)
    print('Averaging operation time used:', time.time() - time_start)
    proba_mat = average_pred['proba'].to_numpy()[..., np.newaxis]
    average_pred['proba'] = minmax_scale(proba_mat[:, 0], feature_range=(0, 1))
    time_start = time.time()
    if threshold is not None:
        average_pred = average_pred[average_pred['proba'] >= threshold]
    elif percentile is not None:
        threshold = np.percentile(average_pred['proba'], percentile)
        average_pred = average_pred[average_pred['proba'] >= threshold]
    else:
        if len(average_pred) > loop_num:
            # threshold = average_pred['proba'].nlargest(loop_num).iloc[-1]
            threshold = np.partition(average_pred['proba'].to_numpy(), -loop_num)[-loop_num]
            # print(threshold)
            original_len = len(average_pred)
            average_pred = average_pred[average_pred['proba'] >= threshold]
            new_len = len(average_pred)
            print('Original length of the average prediction:', original_len)
            # print(new_len / original_len)
    print('Thresholding operation time used:', time.time() - time_start)

    # print(len(candidate_df))
    return average_pred


def read_snap_excel_preds(excel_path, sheet_name):
    df = pd.read_excel(
        excel_path, sheet_name=sheet_name, header=0,
        dtype={'chr1': str, 'chr2': str}, engine='openpyxl'
    )
    df['chr1'] = 'chr' + df['chr1']
    df['chr2'] = 'chr' + df['chr2']
    df = df.rename(columns={'chr1': 'chrom1', 'chr2': 'chrom2'})
    return df


def random_select_cells_from_ds(ds, chrom_names, desired_cell_num, seed):
    np.random.seed(seed)
    random.seed(seed)
    assert len(ds) % len(chrom_names) == 0
    indices = np.random.choice(len(ds) // len(chrom_names), desired_cell_num, replace=False)
    indices = expand_cell_ids_to_graph_ids(indices * len(chrom_names), len(chrom_names))
    small_ds = ds.index_select(indices)
    assert len(small_ds) == len(chrom_names) * desired_cell_num
    return small_ds


