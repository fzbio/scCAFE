import torch


# Inferece time configs
# =======================================================


# Settings for predicting on the mES dataset.
class InferenceConfigs(object):
    pass



# Experiment Settings for predicting on the mES dataset (cross-species).
# Trained on hpc!!!!!
# class ExperimentInferenceConfigs(object):
#     k = 3
#     trained_model_id = 'hpc_multitask'
#     model_dir = 'models'
#     chroms = ['chr' + str(i) for i in range(1, 20)]
#     chrom_sizes_path = 'external_annotations/mm10.sizes'
#
#     motif_feature_path = f'data/graph_features/mouse/CTCF_mm10.10kb.input.csv'
#     kmer_feature_path = f'data/graph_features/mouse/mm10.10kb.kmer.csv'
#
#     raw_finer_scool = 'data/mES/nagano_10kb_filtered.scool'
#     do_imputation = True
#
#     imputed_scool_dir = 'refined_testset_scools'
#
#     filter_region_path = 'region_filter/mm10_filter_regions.txt'
#     bedpe_dict = {
#         'ES': 'data/placeholder'
#     }
#     tad_dict = {
#         'ES': 'data/placeholder'
#     }
#     name_parser = None
#     desired_cell_types = None
#     assembly_path = '/home/fuzhou/hic_research/sc-hic-loop/data/graph_features/mouse/mm10.fa'




# Experiment Settings for predicting on the mES dataset (cross-species).
# Trained on Neuron!!!!!
# class ExperimentInferenceConfigs(object):
#     k = 3
#     trained_model_id = 'neuron_multitask2.5mb'
#     model_dir = 'models'
#     chroms = ['chr' + str(i) for i in range(1, 20)]
#     chrom_sizes_path = 'external_annotations/mm10.sizes'
#
#     motif_feature_path = f'data/graph_features/mouse/CTCF_mm10.10kb.input.csv'
#     kmer_feature_path = f'data/graph_features/mouse/mm10.10kb.kmer.csv'
#
#     raw_finer_scool = 'data/mES/nagano_10kb_filtered.scool'
#     do_imputation = True
#
#     imputed_scool_dir = 'refined_testset_scools'
#
#     filter_region_path = 'region_filter/mm10_filter_regions.txt'
#     bedpe_dict = {
#         'ES': 'data/placeholder'
#     }
#     tad_dict = {
#         'ES': 'data/placeholder'
#     }
#     name_parser = None
#     desired_cell_types = None
#     assembly_path = '/home/fuzhou/hic_research/sc-hic-loop/data/graph_features/mouse/mm10.fa'


# Settings for predicting on the hPFC dataset.
class ExperimentInferenceConfigs(object):
    k = 3
    trained_model_id = 'mES_multitask2.5mb'
    model_dir = 'models'
    chroms = ['chr' + str(i) for i in range(1, 23)]
    chrom_sizes_path = 'external_annotations/hg19.sizes'

    motif_feature_path = f'data/graph_features/human/CTCF_hg19.10kb.input.csv'
    kmer_feature_path = f'data/graph_features/human/hg19.10kb.kmer.csv'

    # raw_finer_scool = 'data/human_prefrontal_cortex/luo_10kb_filtered.scool'
    raw_finer_scool = 'data/human_prefrontal_cortex/luo_10kb_raw.scool'
    do_imputation = False

    imputed_scool_dir = 'refined_testset_scools'

    filter_region_path = 'region_filter/hg19_filter_regions.txt'
    bedpe_dict = {
        'hPFC': 'data/placeholder'
    }
    tad_dict = {
        'hPFC': 'data/placeholder'
    }
    name_parser = None
    desired_cell_types = None

    assembly_path = '/home/fuzhou/hic_research/sc-hic-loop/data/graph_features/human/hg19.fa'




CELL_SELECTION_SEED = 2222
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SelectedHyperparameters(object):
    use_auto_weighted, alpha, beta, a, b, freeze_cross_stitch = False, 0.5, 0.5, 0.9, 0.1, False