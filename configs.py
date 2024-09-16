import torch
from utils import hpc_celltype_parser


# Training time configs
# =======================================================

class CompilationConfigs(object):

    # These are configs for the main model
    latent_channels = 64
    head_input_channels = 64
    learning_rate = 2e-5
    weight_decay = 1e-3
    label_smoothing = 0.0
    gaussian_mean = 0.0
    gaussian_std = 0.0


    # These are configs for the TAD model
    tad_learning_rate = 1e-4
    tad_weight_decay = 1e-3
    tad_window_size = 2000


    # These are configs for the DCN (compartment calling model)
    dcn_lambda = 1.0   # Distance loss coefficient
    dcn_n_clusters = 2
    dcn_input_dim = 2 * head_input_channels
    dcn_latent_dim = 16
    dcn_lr = 1e-4
    dcn_wd = 1e-3



# class TrainConfigs(object):   # Train on GM12878
#     do_imputation = True
#     chrom_sizes_path = 'external_annotations/hg38.chrom.sizes'
#     n_fold = 10
#     chroms = ['chr' + str(i) for i in range(1, 23)]
#     train_chroms = chroms[:12]
#     val_chroms = chroms[12:16]
#     test_chroms = chroms[16:]
#     model_dir = 'models'
#     refined_data_dir = 'refined_scools'
#     graph_dir = 'graph_data'
#
#     finer_scool = 'data/GM12878_low/gm12878_low_tang_hg38_10kb.scool'
#
#     tmp_root_dir = 'tmp'
#
#
#
#     # These are train configs for the main model
#     kmer_feature_path = 'data/graph_features/human/hg38.10kb.kmer.csv'
#     motif_feature_path = 'data/graph_features/human/CTCF_hg38.10kb.input.csv'
#
#     bedpe_dict = {
#         'GM12878': 'data/GM12878_low/GM12878.bedpe'
#     }
#     tad_dict = {
#         'GM12878': 'data/GM12878_low/bulk/GM12878_bulk_hg38_10kb.norm.no_log.imputed.insulation_70kb.bed'
#     }
#     # name_parser = hpc_celltype_parser
#     name_parser = None
#     # desired_cell_types = ['MG', 'ODC', 'Neuron']
#     desired_cell_types = None
#
#     epochs = 100   # Should be 100
#     train_samples_per_epoch = 100   # Should be 100
#
#
#     # These are train configs for the DCN (compartment calling model)
#     dcn_pretrain = True
#     dcn_pretrain_epochs = 50
#     dcn_batch_size = 1
#     dcn_epochs = 50
#     dcn_train_samples_per_epoch = 100
#     dcn_save_best_model = True


# class TrainConfigs(object):   # Train on Neuron, WHOLE settings.
#
#     do_imputation = True
#     chrom_sizes_path = 'external_annotations/hg19.sizes'
#
#
#     n_fold = 10
#     train_chroms = ['chr' + str(i) for i in range(2, 15)] + ['chr' + str(i) for i in range(16, 19)]  # Chromosome 1 and 15 broken for unknown reason
#
#     val_chroms = ['chr' + str(i) for i in range(19, 23)]
#     chroms = train_chroms + val_chroms
#     model_dir = 'models'
#     refined_data_dir = 'refined_scools'
#     graph_dir = 'graph_data'
#
#     finer_scool = 'data/Neuron/Neuron_hg19_10kb_filtered.scool'
#
#     tmp_root_dir = 'tmp'
#
#
#
#     # These are train configs for the main model
#     kmer_feature_path = 'data/graph_features/human/hg19.10kb.kmer.csv'
#     motif_feature_path = 'data/graph_features/human/CTCF_hg19.10kb.input.csv'
#
#     bedpe_dict = {
#         'Neuron': 'data/Neuron/Neuron.bedpe'
#     }
#     tad_dict = {
#         'Neuron': 'data/human_prefrontal_cortex/bulk/neuron_liftedHg19_10kb_directionality_2.5mb.bed'
#     }
#     name_parser = None
#     # name_parser = None
#     desired_cell_types = None
#     # desired_cell_types = ['MG', 'ODC', 'Neuron']
#     # desired_cell_types = None
#
#     epochs = 100   # Should be 100
#     train_samples_per_epoch = 100   # Should be 100
#
#
#     # Train configs for the TAD model
#     tad_samples_per_epoch = 100
#
#
#     # These are train configs for the DCN (compartment calling model)
#     dcn_batch_size = 1


class TrainConfigs(object):   # Train on mES, WHOLE settings.
    do_imputation = True
    chrom_sizes_path = 'external_annotations/mm10.sizes'
    n_fold = 10
    chroms = ['chr' + str(i) for i in range(1, 20)]
    train_chroms = chroms[:16]
    val_chroms = chroms[16:]
    model_dir = 'models'
    refined_data_dir = 'refined_scools'
    graph_dir = 'graph_data'

    finer_scool = 'data/mES/nagano_10kb_filtered.scool'

    tmp_root_dir = 'tmp'



    # These are train configs for the main model
    kmer_feature_path = 'data/graph_features/mouse/mm10.10kb.kmer.csv'
    motif_feature_path = 'data/graph_features/mouse/CTCF_mm10.10kb.input.csv'

    bedpe_dict = {
        # 'MG': 'data/human_prefrontal_cortex/MG.bedpe', 'Neuron': 'data/human_prefrontal_cortex/Neuron.bedpe',
        # 'ODC': 'data/human_prefrontal_cortex/ODC.bedpe'
        'ES': 'data/mES/ES.bedpe'
    }
    tad_dict = {
        'ES': 'data/mES/bulk/mES_mm10_10kb_directionality_2.5mb.bed'
    }
    # name_parser = hpc_celltype_parser
    name_parser = None
    # desired_cell_types = ['MG', 'ODC', 'Neuron']
    desired_cell_types = None

    epochs = 100   # Should be 100
    train_samples_per_epoch = 100   # Should be 100

    # Train configs for the TAD model
    tad_samples_per_epoch = 100


    # These are train configs for the DCN (compartment calling model)
    dcn_batch_size = 1


# class TrainConfigs(object):   # Train on hPFC, WHOLE settings. Mix of 3 cell types. Two types of cells do not have TAD labels.
#
#     do_imputation = True
#     chrom_sizes_path = 'external_annotations/hg19.sizes'
#
#
#     n_fold = 10
#     train_chroms = ['chr' + str(i) for i in range(2, 15)] + ['chr' + str(i) for i in range(16, 19)]  # Chromosome 1 and 15 broken for unknown reason
#
#     val_chroms = ['chr' + str(i) for i in range(19, 23)]
#     chroms = train_chroms + val_chroms
#     model_dir = 'models'
#     refined_data_dir = 'refined_scools'
#     graph_dir = 'graph_data'
#
#     finer_scool = 'data/human_prefrontal_cortex/luo_10kb_filtered.scool'
#
#     tmp_root_dir = 'tmp'
#
#
#
#     # These are train configs for the main model
#     kmer_feature_path = 'data/graph_features/human/hg19.10kb.kmer.csv'
#     motif_feature_path = 'data/graph_features/human/CTCF_hg19.10kb.input.csv'
#
#     bedpe_dict = {
#         'Neuron': 'data/human_prefrontal_cortex/Neuron.bedpe',
#         'MG': 'data/human_prefrontal_cortex/MG.bedpe',
#         'ODC': 'data/human_prefrontal_cortex/ODC.bedpe'
#     }
#     tad_dict = {
#         'Neuron': 'data/human_prefrontal_cortex/bulk/neuron_bulk_liftedHg19_10kb.norm.no_log.imputed.insulation_70kb.bed',
#         'MG': 'data/placeholder',
#         'ODC': 'data/placeholder'
#     }
#     # name_parser = None
#     name_parser = hpc_celltype_parser
#     # desired_cell_types = None
#     desired_cell_types = ['MG', 'ODC', 'Neuron']
#     # desired_cell_types = None
#
#     epochs = 100   # Should be 100
#     train_samples_per_epoch = 100   # Should be 100
#
#     # Train configs for the TAD model
#     tad_samples_per_epoch = 100
#
#
#     # These are train configs for the DCN (compartment calling model)

#     dcn_batch_size = 1



# Torch loader configs
LOADER_WORKER = 3




# Shared configs; users should not change these
# =======================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
DATA_SPLIT_SEED = 2222
SEED = 2222
