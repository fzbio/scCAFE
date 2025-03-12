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



# Torch loader configs
LOADER_WORKER = 3




# Shared configs; users should not change these
# =======================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
DATA_SPLIT_SEED = 2222
SEED = 2222
