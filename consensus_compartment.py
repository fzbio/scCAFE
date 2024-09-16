from nn_data import StreamScoolDataset, expand_cell_ids_to_graph_ids, ReadKmerFeatures, ReadMotifFeatures, PositionalEncoding
from torch_geometric import transforms as T
from nn_data import RemoveSelfLooping
import os
from multiscale_calling import CrossStitchFeatureCaller, MultitaskFeatureCaller
from inference_configs import SelectedHyperparameters as hyperparams
from schickit.utils import get_chrom_sizes, get_bin_count
from middleware import MiddleWareDataset
from compartment import CompartmentCaller
from inference_configs import ExperimentInferenceConfigs as eic, CELL_SELECTION_SEED
import cooler
import tempfile
import argparse
from schickit.data_storage import random_select_subset_scools
import glob


# Consensus compartment on mES
if __name__ == '__main__':
    valid_chroms = ['chr' + str(i) for i in range(1, 20)]
    chrom_sizes = get_chrom_sizes('external_annotations/mm10.sizes')
    chrom_sizes = {chrom: chrom_sizes[chrom] for chrom in valid_chroms}

    compartment_preds_dir = 'preds/neuron_multitask2.5mb_test.on.mES_742_replicate0/compartment_preds'
    all_compartment_preds_paths = glob.glob(f'{compartment_preds_dir}/*.csv')

    for cell_type in ['mES']:
        cell_type_compartment_preds_paths = [path for path in all_compartment_preds_paths]
        output_path = f'preds/consensus_compartment/consensus_compartment_on_{cell_type}.bed'
        compartment_caller = CompartmentCaller()
        compartment_caller.consensus_compartments(cell_type_compartment_preds_paths, output_path)



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('-d', '--use-data', action='store_true')
#     args = parser.parse_args()
#     use_existing_data = args.use_data
#     valid_chroms = eic.chroms
#     cell_num = 742
#     eval_time = 0
#     pred_id = f'{eic.trained_model_id}_test.on.mES_{cell_num}_replicate{eval_time}'
#     chrom_sizes = get_chrom_sizes(eic.chrom_sizes_path)
#     chrom_sizes = {chrom: chrom_sizes[chrom] for chrom in valid_chroms}
#
#     tmp_root = 'tmp'
#
#     with tempfile.TemporaryDirectory(dir=tmp_root) as graph_dir, \
#             tempfile.TemporaryDirectory(dir=tmp_root) as subset_dir, \
#             tempfile.TemporaryDirectory(dir=tmp_root) as trainset_dir, tempfile.TemporaryDirectory(dir=tmp_root) as valset_dir:
#
#         if use_existing_data:
#             imputed_finer_scool_path = os.path.join(eic.imputed_scool_dir, f'{pred_id}.scool')
#         else:
#             selected_raw_finer_scool_path = os.path.join(subset_dir, 'subset.scool')
#             random_select_subset_scools(eic.raw_finer_scool, selected_raw_finer_scool_path, cell_num,
#                                         CELL_SELECTION_SEED)
#             assert len(cooler.fileops.list_scool_cells(selected_raw_finer_scool_path)) == cell_num
#             imputed_finer_scool_path = selected_raw_finer_scool_path
#         assert len(cooler.fileops.list_scool_cells(imputed_finer_scool_path)) == cell_num
#
#
#         for cell_type in ['mES']:
#             graph_dataset = StreamScoolDataset(
#                 graph_dir,
#                 imputed_finer_scool_path,
#                 valid_chroms, 10000,
#                 eic.bedpe_dict, eic.tad_dict,
#                 lambda x, y: 'ES', ['ES'],
#                 pre_transform=T.Compose([
#                     RemoveSelfLooping(),
#                     ReadKmerFeatures(
#                         eic.kmer_feature_path, valid_chroms, False,
#                         os.path.join(eic.model_dir, f'{eic.trained_model_id}_kmer_scaler_calling.pkl')
#                     ),
#                     ReadMotifFeatures(
#                         eic.motif_feature_path, valid_chroms, False,
#                         os.path.join(eic.model_dir, f'{eic.trained_model_id}_motif_scaler_calling.pkl')
#                     ),
#                     PositionalEncoding()
#                 ])
#             )
#             feature_caller = MultitaskFeatureCaller(
#                 eic.trained_model_id, valid_chroms, f'{eic.model_dir}/{eic.trained_model_id}.pt',
#                 graph_dataset.num_features,
#                 hyperparams.alpha, hyperparams.beta
#             )
#             feature_caller.load_model()
#             test_set = MiddleWareDataset(graph_dataset, feature_caller)
#
#
#             print(f'Predicting consensus compartment {cell_type}...')
#             output_path = f'preds/consensus_compartment/consensus_compartment_on_{cell_type}.bed'
#             compartment_caller = CompartmentCaller()
#             compartment_caller.consensus_compartments(test_set, valid_chroms, eic.assembly_path, eic.chrom_sizes_path, 10000, output_path, lambda x: True)