# This program is used locally (not released to public).

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
from utils import json_to_object
import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict loops on a single-cell Hi-C dataset.')
    parser.add_argument('config_path', type=str, help='Path to the configuration file.')
    parser.add_argument('pred_id', type=str, help='User self-defined, unique ID of the prediction.')
    parser.add_argument(
        '-d', '--use-data', action='store_true',
        help='Use existing, already enhanced data. Set this to true only when you set `imputation` '
             'to true in the config file and have already run one of the inference scripts. '
             'Default: False.'
    )
    args = parser.parse_args()
    config_path = args.config_path
    pred_id = args.pred_id
    assert os.path.isfile(config_path)
    eic = json_to_object(config_path)
    use_existing_data = args.use_data

    cell_selection_seed = SEED
    run_id = eic.trained_model_id
    cell_selection_seed = cell_selection_seed
    chroms = eic.chroms
    model_dir = eic.model_dir
    tmp_root = 'tmp'
    pred_output_dir = f'preds/{pred_id}'
    loop_dir = os.path.join(pred_output_dir, 'loop_preds')
    tad_dir = os.path.join(pred_output_dir, 'tad_preds')
    filtered_loop_dir = os.path.join(pred_output_dir, 'loop_preds_filtered')
    bedpe_dict = eic.bedpe_dict
    os.makedirs('preds/', exist_ok=True)
    placeholder_path = 'data/placeholder'
    with open(placeholder_path, "w") as file:
        pass
    tad_dict = {k: placeholder_path for k in bedpe_dict}
    kmer_feature_path = eic.kmer_feature_path
    motif_feature_path = eic.motif_feature_path
    chrom_sizes_path = eic.chrom_sizes_path
    name_parser = None
    desired_cell_types = None
    save_to_hdf5 = eic.save_to_hdf if hasattr(eic, 'save_to_hdf') else False
    predict_all = False

    cell_num = len(cooler.fileops.list_scool_cells(eic.raw_finer_scool))

    with tempfile.TemporaryDirectory(dir=tmp_root) as graph_dir, \
            tempfile.TemporaryDirectory(dir=tmp_root) as subset_dir:
        selected_raw_finer_scool_path = os.path.join(subset_dir, 'subset.scool')
        random_select_subset_scools(eic.raw_finer_scool, selected_raw_finer_scool_path, cell_num, cell_selection_seed)
        assert len(cooler.fileops.list_scool_cells(selected_raw_finer_scool_path)) == cell_num
        if eic.do_imputation:
            imputed_finer_scool_path = os.path.join(eic.imputed_scool_dir, f'{pred_id}.scool')
            if not use_existing_data:
                remove_existing_scool(imputed_finer_scool_path)
                assembly = get_chrom_sizes(chrom_sizes_path)
                imputer = Imputer(3)
                imputer.impute_dataset(selected_raw_finer_scool_path, imputed_finer_scool_path, chroms, assembly, tmp_root)
        else:
            imputed_finer_scool_path = selected_raw_finer_scool_path
        assert len(cooler.fileops.list_scool_cells(imputed_finer_scool_path)) == cell_num


        embedding_dir = None

        lower_upper_bound = (10, 100) if predict_all else None
        predict_on_other_dataset(
            model_dir, run_id, chroms, bedpe_dict, tad_dict, imputed_finer_scool_path, graph_dir, 0.0,
            loop_dir, kmer_feature_path, motif_feature_path, predict_all, lower_upper_bound,
            name_parser, desired_cell_types, embedding_dir, save_to_hdf5
        )

    if eic.filter_region_path is not None:
        processor = PostProcessor()
        processor.read_filter_file(eic.filter_region_path)
        processor.remove_invalid_loops_in_dir(
            loop_dir, filtered_loop_dir, proba_threshold=0.0
        )