from multiscale_calling import MultitaskFeatureCaller
from configs import DEVICE, SEED, TrainConfigs as tc, CompilationConfigs as cc
import torch
from gnns import NodeNN
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler
from train_utils import EarlyStopper
from tqdm.auto import tqdm
import numpy as np
from torchmetrics.classification import BinaryAccuracy
import os
from configs import LOADER_WORKER
import pandas as pd
from schickit.utils import create_bin_df
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale


def create_linear_connectivity_matrix(n):
    A = lil_matrix((n, n), dtype=int)

    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1

    return A


def create_coassociation_matrix(cluster_results):
    n_samples = len(cluster_results[0])
    coassoc_matrix = np.zeros((n_samples, n_samples))

    for clusters in tqdm(cluster_results):
        clusters = np.array(clusters)
        same_cluster = clusters[:, np.newaxis] == clusters[np.newaxis, :]
        coassoc_matrix += same_cluster

    return coassoc_matrix / len(cluster_results)


# def add_pseudo_node_to_adj(coo_adj):
#     # Add a pseudo node to the coo_matrix
#     n = coo_adj.shape[0]
#     original_data = coo_adj.data
#     original_row = coo_adj.row
#     original_col = coo_adj.col
#     new_data = np.concatenate([original_data, np.ones((2 * n,))])
#     new_row = np.concatenate([original_row, np.ones((n,)) * n, np.arange(n)])
#     new_col = np.concatenate([original_col, np.arange(n), np.ones((n,)) * n])
#     return coo_matrix((new_data, (new_row, new_col)), shape=(n + 1, n + 1))

def add_pseudo_node_to_adj(adj):
    # Add a pseudo node to the dense matrix
    n = adj.shape[0]
    new_adj = np.zeros((n + 1, n + 1))
    new_adj[:n, :n] = adj
    new_adj[n, :n] = 1
    new_adj[:n, n] = 1
    return new_adj


def convert_segmentation_to_break_points(segmentation):
    change_points = []
    current_label = segmentation[0]

    for i in range(1, len(segmentation)):
        if segmentation[i] != current_label:
            change_points.append(i)
            current_label = segmentation[i]

    return np.array(change_points)


def get_segments_with_higher_ctcf(z, orig_segment, ctcf_vec):
    assert len(z) == len(ctcf_vec)
    change_points = convert_segmentation_to_break_points(orig_segment)
    z_change_points = z[change_points]
    # clusterer = KMeans(n_clusters=2, n_init='auto')
    clusterer = AgglomerativeClustering(n_clusters=2)
    # clusterer = GaussianMixture(n_components=2)
    pred = clusterer.fit_predict(z_change_points)

    cluster_ctcf = []
    for c in np.sort(np.unique(pred)):
        cluster_ctcf.append(np.mean(ctcf_vec[change_points[pred == c]]))
    highest_ctcf_cluster = np.argmax(cluster_ctcf)
    change_points = change_points[pred == highest_ctcf_cluster]
    new_segment = np.zeros_like(orig_segment, dtype=int)
    for i, change_point in enumerate(change_points):
        if i == 0:
            new_segment[:change_point] = i
        elif i == len(change_points) - 1:
            new_segment[change_point:] = i + 1
            new_segment[change_points[i - 1]:change_point] = i
        else:
            new_segment[change_points[i-1]:change_point] = i
    return new_segment


class TadCaller(object):
    def __init__(self, ctcf_path=None):
        self.ctcf_path = ctcf_path

    @torch.no_grad()
    def predict(self, tad_out_dir, test_set, device, chrom_sizes, resolution):
        bs = 1
        os.makedirs(tad_out_dir, exist_ok=False)
        loader = DataLoader(test_set, bs, num_workers=0, pin_memory=False, )
        ctcf_df = pd.read_csv(self.ctcf_path, sep='\t', header=0, index_col=False)

        print('Predicting...')
        for i, batch in enumerate(tqdm(loader)):
            z = batch.x.detach().cpu().numpy()
            num_clusters = z.shape[0] // 20   # current best 20
            connectivity = create_linear_connectivity_matrix(z.shape[0])
            z = PCA(n_components=8, whiten=True).fit_transform(z) # current best 4
            # z = FastICA(n_components=4, whiten='arbitrary-variance').fit_transform(z) # current best 4
            clusterer = AgglomerativeClustering(n_clusters=num_clusters, connectivity=connectivity, metric='euclidean', linkage='complete')
            pred = clusterer.fit_predict(z)

            current_ctcf_df = ctcf_df[ctcf_df['chrom'] == batch.chrom_name[0]]
            ctcf_vec = current_ctcf_df['pos_count'].values + current_ctcf_df['neg_count'].values

            pred = get_segments_with_higher_ctcf(z, pred, ctcf_vec)

            df = self.convert_batch_tad_preds_to_df(pred, batch.chrom_name[0], chrom_sizes, resolution)

            short_cell_name = batch.cell_name[0].split('/')[-1]
            cell_csv_path = os.path.join(tad_out_dir, f'{short_cell_name}.csv')
            df.to_csv(
                cell_csv_path, sep='\t', header=not os.path.exists(cell_csv_path),
                index=False, mode='a', float_format='%.5f'
            )
        print('Done!')

    def convert_batch_tad_preds_to_df(self, pred, chrom_name, chrom_sizes, resolution):
        df = create_bin_df(chrom_sizes, resolution, [chrom_name])
        assert len(df) == len(pred)
        df['score'] = pred
        df = df.reset_index(drop=True)
        return df

    def consensus_tads(self, sc_pred_paths, valid_chrom_sizes, out_path):
        dfs = []
        for cell_csv_path in sc_pred_paths:
            df = pd.read_csv(cell_csv_path, sep='\t', header=0, index_col=False)
            dfs.append(df)
        consensus_preds = []
        for chrom in valid_chrom_sizes:
            chrom_dfs = [df[df['chrom'] == chrom].reset_index(drop=True) for df in dfs]
            coass = create_coassociation_matrix([df['score'].values for df in chrom_dfs])
            num_clusters = coass.shape[0] // 100
            coass = add_pseudo_node_to_adj(coass)

            x = SpectralEmbedding(n_components=16, affinity='precomputed').fit_transform(coass)
            x = x[:-1, :]
            connectivity = create_linear_connectivity_matrix(x.shape[0])
            clusterer = AgglomerativeClustering(n_clusters=num_clusters, connectivity=connectivity, metric='euclidean', linkage='complete')
            pred = clusterer.fit_predict(x)
            chrom_df = create_bin_df(valid_chrom_sizes, 10000, [chrom])
            chrom_df['score'] = pred
            consensus_preds.append(chrom_df)

        consensus_df = pd.concat(consensus_preds).reset_index(drop=True)
        consensus_df.to_csv(out_path, sep='\t', index=False)
        return consensus_df


