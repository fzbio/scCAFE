#!/usr/bin/env python3
import os.path

from torch_geometric import transforms as T
from nn_data import RemoveSelfLooping, PositionalEncoding, ReadKmerFeatures, ReadMotifFeatures, GaussianNoise
# from loop_calling import GnnLoopCaller, estimate_chrom_loop_num_train
from multiscale_calling import CrossStitchFeatureCaller
import numpy as np
import torch
from nn_data import StreamScoolDataset, get_train_val_scools
import random
from utils import hpc_celltype_parser, get_split_scool_paths, get_loop_calling_dataset_paths, remove_datasets
import configs
from configs import TrainConfigs as tc
from configs import CompilationConfigs as cc
import sys
from distutils.util import strtobool
from imputation import Imputer
import tempfile
from schickit.utils import get_chrom_sizes
import argparse
from multiscale_calling import MultitaskFeatureCaller


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('seed_shift', type=int)
    parser.add_argument('K', type=int)
    parser.add_argument('run_id', type=str)
    parser.add_argument('-d', '--use-existing-data', action='store_true')
    args = parser.parse_args()
    seed_shift = args.seed_shift  # Different from the cv pipeline, we do not need to train multiple times.
    K = args.K
    run_id = args.run_id   # Get the ID from the command line, so that the script is more flexibly controlled by the user.
    use_existing_data = args.use_existing_data

    alpha = 0.5
    beta = 0.5

    SEED = configs.SEED + seed_shift   # Different from the cv pipeline, here we use a shifted seed to initialize the model. Only run once.
    DATA_SPLIT_SEED = configs.DATA_SPLIT_SEED + seed_shift  # Different from the cv pipeline, here we use a shifted seed to split the dataset into train and val (no test). Only run once.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    raw_finer_scool_name = f'{run_id}_finer_raw'
    imputed_finer_scool_name = f'{run_id}_finer_imputed'

    # Pseudo Imputing starts here (We do not actually impute the data, we just use the raw data as input to the model.)
    if tc.do_imputation:
        imputer = Imputer(K)
        imputed_finer_scool_train, imputed_finer_scool_val, _ = get_split_scool_paths(
            tc.refined_data_dir, imputed_finer_scool_name
        )
        if not use_existing_data:
            with tempfile.TemporaryDirectory(dir=tc.tmp_root_dir) as raw_split_scool_dir:
                raw_finer_scool_train, raw_finer_scool_val, _ = get_split_scool_paths(
                    raw_split_scool_dir, raw_finer_scool_name
                )
                get_train_val_scools(
                    tc.finer_scool, 0.05, DATA_SPLIT_SEED,
                    raw_finer_scool_train, raw_finer_scool_val
                )
                assembly = get_chrom_sizes(tc.chrom_sizes_path)
                imputer.impute_dataset(raw_finer_scool_train, imputed_finer_scool_train, tc.train_chroms, assembly, tc.tmp_root_dir)
                imputer.impute_dataset(raw_finer_scool_val, imputed_finer_scool_val, tc.val_chroms, assembly, tc.tmp_root_dir)
    else:
        imputed_finer_scool_train, imputed_finer_scool_val, _ = get_split_scool_paths(
            tc.refined_data_dir, imputed_finer_scool_name
        )
        if not use_existing_data:
            get_train_val_scools(
                tc.finer_scool, 0.05, DATA_SPLIT_SEED,
                imputed_finer_scool_train, imputed_finer_scool_val
            )

    # Loop calling configs
    loop_calling_dataset_name = f'{run_id}_graph'
    loop_calling_train, loop_calling_val, _ = \
        get_loop_calling_dataset_paths(tc.graph_dir, loop_calling_dataset_name)
    # Feature calling starts here
    # chroms = ['chr21', 'chr22']
    train_set = StreamScoolDataset(
        loop_calling_train,
        imputed_finer_scool_train,
        tc.train_chroms, 10000, tc.bedpe_dict, tc.tad_dict,
        tc.name_parser, tc.desired_cell_types,
        pre_transform=T.Compose([
            RemoveSelfLooping(),
            ReadKmerFeatures(
                tc.kmer_feature_path, tc.train_chroms, True, os.path.join(tc.model_dir, f'{run_id}_kmer_scaler_calling.pkl')
            ),
            ReadMotifFeatures(
                tc.motif_feature_path, tc.train_chroms, True, os.path.join(tc.model_dir, f'{run_id}_motif_scaler_calling.pkl')
            ),
            PositionalEncoding(),
            GaussianNoise(mean=cc.gaussian_mean, std=cc.gaussian_std)
        ])
    )
    val_set = StreamScoolDataset(
        loop_calling_val,
        imputed_finer_scool_val,
        tc.val_chroms, 10000, tc.bedpe_dict, tc.tad_dict,
        tc.name_parser, tc.desired_cell_types,
        pre_transform=T.Compose([
            RemoveSelfLooping(),
            ReadKmerFeatures(
                tc.kmer_feature_path, tc.val_chroms, False, os.path.join(tc.model_dir, f'{run_id}_kmer_scaler_calling.pkl')
            ),
            ReadMotifFeatures(
                tc.motif_feature_path, tc.val_chroms, False, os.path.join(tc.model_dir, f'{run_id}_motif_scaler_calling.pkl')
            ),
            PositionalEncoding()
        ])
    )
    assert len(train_set.chrom_names) == len(tc.train_chroms)
    assert len(train_set) > len(val_set)

    feature_caller = MultitaskFeatureCaller(
        run_id, tc.chroms, f'{tc.model_dir}/{run_id}.pt', train_set.num_features,
        alpha, beta
    )
    feature_caller.train(train_set, val_set, epochs=tc.epochs)
    remove_datasets([loop_calling_train, loop_calling_val])
