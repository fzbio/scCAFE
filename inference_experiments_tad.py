# This program is used locally (not released to public).

from inference_configs import ExperimentInferenceConfigs as eic
from inference_configs import CELL_SELECTION_SEED as SEED
import os
from predict_eval import predict_tads_on_other_dataset
from post_process import PostProcessor
from schickit.data_storage import random_select_subset_scools
from schickit.utils import get_chrom_sizes
import sys
import cooler
import tempfile
from imputation import Imputer
import argparse
from utils import remove_existing_scool

# Predicting on the another schic dataset (mES), with randomly select a certain number of cells
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('eval_time', type=int)
#     parser.add_argument('cell_num', type=int)
#     parser.add_argument('-d', '--use-data', action='store_true')
#     args = parser.parse_args()
#     eval_time = args.eval_time
#     cell_num = args.cell_num
#     use_existing_data = args.use_data
#
#     cell_selection_seed = SEED
#     run_id = eic.trained_model_id
#     cell_selection_seed = cell_selection_seed + eval_time
#     chroms = eic.chroms
#     model_dir = eic.model_dir
#     tmp_root = 'tmp'
#     pred_id = f'{run_id}_test.on.mES_{cell_num}_replicate{eval_time}'
#     pred_output_dir = f'preds/{run_id}_test.on.mES_{cell_num}_replicate{eval_time}'
#     tad_dir = os.path.join(pred_output_dir, 'tad_preds')
#     bedpe_dict = eic.bedpe_dict
#     tad_dict = eic.tad_dict
#     kmer_feature_path = eic.kmer_feature_path
#     motif_feature_path = eic.motif_feature_path
#     chrom_sizes_path = eic.chrom_sizes_path
#     name_parser = eic.name_parser
#     desired_cell_types = eic.desired_cell_types
#
#     with tempfile.TemporaryDirectory(dir=tmp_root) as graph_dir, \
#             tempfile.TemporaryDirectory(dir=tmp_root) as subset_dir, \
#             tempfile.TemporaryDirectory(dir=tmp_root) as trainset_dir, tempfile.TemporaryDirectory(dir=tmp_root) as valset_dir:
#         if use_existing_data:
#             imputed_finer_scool_path = os.path.join(eic.imputed_scool_dir, f'{pred_id}.scool')
#         else:
#             selected_raw_finer_scool_path = os.path.join(subset_dir, 'subset.scool')
#             random_select_subset_scools(eic.raw_finer_scool, selected_raw_finer_scool_path, cell_num,
#                                         cell_selection_seed)
#             assert len(cooler.fileops.list_scool_cells(selected_raw_finer_scool_path)) == cell_num
#             imputed_finer_scool_path = selected_raw_finer_scool_path
#         assert len(cooler.fileops.list_scool_cells(imputed_finer_scool_path)) == cell_num
#
#         predict_tads_on_other_dataset(
#             model_dir, run_id, chroms, bedpe_dict, tad_dict, imputed_finer_scool_path, graph_dir,
#             tad_dir, kmer_feature_path, motif_feature_path, chrom_sizes_path, 10000,
#             name_parser, desired_cell_types
#         )


# Predict on the another schic dataset (hPFC)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('eval_time', type=int)
    parser.add_argument('cell_num', type=int)
    parser.add_argument('-d', '--use-data', action='store_true')
    args = parser.parse_args()
    eval_time = args.eval_time
    cell_num = args.cell_num
    use_existing_data = args.use_data

    cell_selection_seed = SEED
    run_id = eic.trained_model_id
    cell_selection_seed = cell_selection_seed + eval_time
    chroms = eic.chroms
    model_dir = eic.model_dir
    tmp_root = 'tmp'
    pred_id = f'{run_id}_test.on.hPFC_{cell_num}_replicate{eval_time}'
    pred_output_dir = f'preds/{run_id}_test.on.hPFC_{cell_num}_replicate{eval_time}'
    tad_dir = os.path.join(pred_output_dir, 'tad_preds')
    bedpe_dict = eic.bedpe_dict
    tad_dict = eic.tad_dict
    kmer_feature_path = eic.kmer_feature_path
    motif_feature_path = eic.motif_feature_path
    chrom_sizes_path = eic.chrom_sizes_path
    name_parser = eic.name_parser
    desired_cell_types = eic.desired_cell_types

    with tempfile.TemporaryDirectory(dir=tmp_root) as graph_dir, \
            tempfile.TemporaryDirectory(dir=tmp_root) as subset_dir, \
            tempfile.TemporaryDirectory(dir=tmp_root) as trainset_dir, tempfile.TemporaryDirectory(dir=tmp_root) as valset_dir:
        if use_existing_data:
            imputed_finer_scool_path = os.path.join(eic.imputed_scool_dir, f'{pred_id}.scool')
        else:
            selected_raw_finer_scool_path = os.path.join(subset_dir, 'subset.scool')
            random_select_subset_scools(eic.raw_finer_scool, selected_raw_finer_scool_path, cell_num,
                                        cell_selection_seed)
            assert len(cooler.fileops.list_scool_cells(selected_raw_finer_scool_path)) == cell_num
            imputed_finer_scool_path = selected_raw_finer_scool_path
        assert len(cooler.fileops.list_scool_cells(imputed_finer_scool_path)) == cell_num

        predict_tads_on_other_dataset(
            model_dir, run_id, chroms, bedpe_dict, tad_dict, imputed_finer_scool_path, graph_dir,
            tad_dir, kmer_feature_path, motif_feature_path, chrom_sizes_path, 10000,
            name_parser, desired_cell_types
        )
