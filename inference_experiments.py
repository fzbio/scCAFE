# This program is used locally (not released to public).

from inference_configs import ExperimentInferenceConfigs as eic
from inference_configs import CELL_SELECTION_SEED as SEED
import os
from predict_eval import predict_on_other_dataset
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
#     loop_dir = os.path.join(pred_output_dir, 'loop_preds')
#     tad_dir = os.path.join(pred_output_dir, 'tad_preds')
#     filtered_loop_dir = os.path.join(pred_output_dir, 'loop_preds_filtered')
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
#         selected_raw_finer_scool_path = os.path.join(subset_dir, 'subset.scool')
#         random_select_subset_scools(eic.raw_finer_scool, selected_raw_finer_scool_path, cell_num, cell_selection_seed)
#         assert len(cooler.fileops.list_scool_cells(selected_raw_finer_scool_path)) == cell_num
#         if eic.do_imputation:
#             imputed_finer_scool_path = os.path.join(eic.imputed_scool_dir, f'{pred_id}.scool')
#             if not use_existing_data:
#                 remove_existing_scool(imputed_finer_scool_path)
#                 assembly = get_chrom_sizes(chrom_sizes_path)
#                 imputer = Imputer(eic.k)
#                 imputer.impute_dataset(selected_raw_finer_scool_path, imputed_finer_scool_path, chroms, assembly, tmp_root)
#         else:
#             imputed_finer_scool_path = selected_raw_finer_scool_path
#         assert len(cooler.fileops.list_scool_cells(imputed_finer_scool_path)) == cell_num
#
#         predict_on_other_dataset(
#             model_dir, run_id, chroms, bedpe_dict, tad_dict, imputed_finer_scool_path, graph_dir, 0.0,
#             loop_dir, tad_dir, kmer_feature_path, motif_feature_path, chrom_sizes_path,
#             name_parser, desired_cell_types
#         )
#
#     processor = PostProcessor()
#     processor.read_filter_file(eic.filter_region_path)
#     processor.remove_invalid_loops_in_dir(
#         loop_dir, filtered_loop_dir, proba_threshold=0.0
#     )


# Predict on the another schic dataset (hPFC)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('eval_time', type=int)
    parser.add_argument('cell_num', type=int)
    parser.add_argument('-d', '--use-data', action='store_true')
    parser.add_argument('-e', '--embedding', action='store_true')
    args = parser.parse_args()
    eval_time = args.eval_time
    cell_num = args.cell_num
    use_existing_data = args.use_data
    generate_embedding = args.embedding

    cell_selection_seed = SEED
    run_id = eic.trained_model_id
    cell_selection_seed = cell_selection_seed + eval_time
    chroms = eic.chroms
    model_dir = eic.model_dir
    tmp_root = 'tmp'
    pred_id = f'{run_id}_test.on.hPFC_{cell_num}_replicate{eval_time}'
    pred_output_dir = f'preds/{run_id}_test.on.hPFC_{cell_num}_replicate{eval_time}'
    # pred_output_dir = f'/mnt/d/research_projects/scCAFE_data/preds/{run_id}_test.on.hPFC_{cell_num}_replicate{eval_time}'
    loop_dir = os.path.join(pred_output_dir, 'loop_preds')
    tad_dir = os.path.join(pred_output_dir, 'tad_preds')
    filtered_loop_dir = os.path.join(pred_output_dir, 'loop_preds_filtered')
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
        selected_raw_finer_scool_path = os.path.join(subset_dir, 'subset.scool')
        random_select_subset_scools(eic.raw_finer_scool, selected_raw_finer_scool_path, cell_num, cell_selection_seed)
        assert len(cooler.fileops.list_scool_cells(selected_raw_finer_scool_path)) == cell_num
        if eic.do_imputation:
            imputed_finer_scool_path = os.path.join(eic.imputed_scool_dir, f'{pred_id}.scool')
            if not use_existing_data:
                remove_existing_scool(imputed_finer_scool_path)
                assembly = get_chrom_sizes(chrom_sizes_path)
                imputer = Imputer(eic.k)
                imputer.impute_dataset(selected_raw_finer_scool_path, imputed_finer_scool_path, chroms, assembly, tmp_root)
        else:
            imputed_finer_scool_path = selected_raw_finer_scool_path
        assert len(cooler.fileops.list_scool_cells(imputed_finer_scool_path)) == cell_num

        if generate_embedding:
            embedding_dir = os.path.join(pred_output_dir, 'embeddings')
        else:
            embedding_dir = None

        predict_on_other_dataset(
            model_dir, run_id, chroms, bedpe_dict, tad_dict, imputed_finer_scool_path, graph_dir, 0.0,
            loop_dir, tad_dir, kmer_feature_path, motif_feature_path, chrom_sizes_path,
            name_parser, desired_cell_types, embedding_dir
        )

    processor = PostProcessor()
    processor.read_filter_file(eic.filter_region_path)
    processor.remove_invalid_loops_in_dir(
        loop_dir, filtered_loop_dir, proba_threshold=0.0
    )