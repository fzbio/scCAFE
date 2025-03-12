import torch
from scipy.spatial.distance import euclidean
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
from sympy.physics.vector.tests.test_printing import alpha

from gnns import NodeNN
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler
from train_utils import EarlyStopper
from tqdm.auto import tqdm
import numpy as np
from torchmetrics.classification import BinaryAccuracy
import os
from configs import LOADER_WORKER, SEED
import pandas as pd
from schickit.utils import create_bin_df
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale
from scipy.stats import wasserstein_distance


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


def convert_segmentation_to_tuples(segmentation, chrom_list):
    segments = []
    start = 0
    current_label = segmentation[0]
    current_chrom = chrom_list[0]

    for i in range(1, len(segmentation)):
        if segmentation[i] != current_label or chrom_list[i] != current_chrom:
            segments.append((start, i))
            start = i
            current_label = segmentation[i]
            current_chrom = chrom_list[i]

    # Append the last segment
    segments.append((start, len(segmentation)))

    return segments


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


def boundary_score(z, pred):
    break_points = convert_segmentation_to_break_points(pred)
    if break_points[0] == 1:
        break_points = break_points[1:]
    z_boundary = z[break_points - 1]
    z_inter = (z[break_points] + z[break_points + 1] + z[break_points + 2] + z[break_points + 3]) / 4
    z_intra = (z[break_points - 2] + z[break_points - 3] + z[break_points - 4] + z[break_points - 5]) / 4
    inter_distances = np.linalg.norm(z_inter - z_boundary, axis=1)
    intra_distances = np.linalg.norm(z_intra - z_boundary, axis=1)
    return np.mean(inter_distances / intra_distances)


def size_uniformaity_score(pred):
    cluster_sizes = np.bincount(pred)
    cluster_sizes = cluster_sizes[cluster_sizes > 0]
    return np.std(cluster_sizes) / np.mean(cluster_sizes)


def ctcf_score(z, pred, ctcf_vec):
    change_points = convert_segmentation_to_break_points(pred)
    ctcf_break_points = np.concatenate([ctcf_vec[change_points], ctcf_vec[change_points - 1]])
    return np.mean(ctcf_break_points)

def remove_invalid_clusters(z, pred, ctcf_vec, min_cluster_size, max_cluster_size):
    cluster_sizes = np.bincount(pred)
    small_clusters = np.where(cluster_sizes < min_cluster_size)[0]
    large_clusters = np.where(cluster_sizes > max_cluster_size)[0]
    invalid_clusters = np.concatenate([small_clusters, large_clusters])
    invalid_clusters = np.unique(invalid_clusters)
    for cluster in invalid_clusters:
        pred[pred == cluster] = -1
    mask = pred != -1
    pred = pred[mask]
    z = z[mask, :]
    ctcf_vec = ctcf_vec[mask]
    return z, pred, ctcf_vec

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def size_distribution_similarity_score(pred, gt_tad_df, resolution, lower_bound, upper_bound):
    gt_distribution = (gt_tad_df['end'] - gt_tad_df['start']) // resolution
    gt_distribution = gt_distribution[(gt_distribution >= lower_bound) & (gt_distribution <= upper_bound)]

    pred_distribution = []
    for cluster in np.unique(pred):
        pred_distribution.append(np.sum(pred == cluster))
    pred_distribution = np.array(pred_distribution)
    if len(pred_distribution) == 0:
        return np.nan
    else:
        dist = wasserstein_distance(gt_distribution, pred_distribution)
        return dist


class TadCaller(object):
    def __init__(self, ctcf_path=None):
        self.ctcf_path = ctcf_path

    def select_best_average_size(self, test_set, ref_tad_path, subset_size=30, plot_path=None):
        potential_average_sizes = np.arange(5, 100)
        valid_average_sizes = []
        bs = 1
        test_set = test_set.index_select(
            torch.tensor(np.random.RandomState(SEED).choice(len(test_set), subset_size, replace=False))
        )
        loader = DataLoader(test_set, bs, num_workers=0, pin_memory=False)
        ctcf_df = pd.read_csv(self.ctcf_path, sep='\t', header=0, index_col=False)
        mean_metric_scores = []
        std_metric_scores = []
        print('Selecting best average size...')
        for size in tqdm(potential_average_sizes):
            current_size_metric_scores = []
            for idx, batch in enumerate(loader):
                z = batch.x.detach().cpu().numpy()
                num_clusters = z.shape[0] // size
                connectivity = create_linear_connectivity_matrix(z.shape[0])
                z = PCA(n_components=8, whiten=True).fit_transform(z)
                clusterer = AgglomerativeClustering(n_clusters=num_clusters, connectivity=connectivity,
                                                    metric='euclidean', linkage='complete')
                pred = clusterer.fit_predict(z)

                current_ctcf_df = ctcf_df[ctcf_df['chrom'] == batch.chrom_name[0]]
                ctcf_vec = current_ctcf_df['pos_count'].values + current_ctcf_df['neg_count'].values
                pred = get_segments_with_higher_ctcf(z, pred, ctcf_vec)
                z, pred, ctcf_vec = remove_invalid_clusters(z, pred, ctcf_vec, 10, 100)
                # current_size_metric_scores.append(boundary_score(z, pred))

                ref_tad_df = pd.read_csv(ref_tad_path, sep='\t', header=0, index_col=False)
                current_score = size_distribution_similarity_score(pred, ref_tad_df, 10000, 10, 100)
                if not np.isnan(current_score):
                    current_size_metric_scores.append(current_score)
            if len(current_size_metric_scores) != 0:
                valid_average_sizes.append(size)
                mean_metric_scores.append(np.mean(current_size_metric_scores))
                std_metric_scores.append(np.std(current_size_metric_scores))
        if plot_path is not None:
            plt.plot(valid_average_sizes, mean_metric_scores, color='#b05454')
            # plt.errorbar(valid_average_sizes, mean_metric_scores, yerr=std_metric_scores)
            plt.fill_between(valid_average_sizes, np.array(mean_metric_scores) - np.array(std_metric_scores),
                                np.array(mean_metric_scores) + np.array(std_metric_scores), alpha=0.25, color='#b05454')
            plt.xlabel('Average TLD size $\langle \mathrm{TLD} \\rangle$')
            plt.ylabel('Dissimilarity score ($\mathcal{D}$)')
            best_average_size = valid_average_sizes[np.argmin(mean_metric_scores)]
            plt.axvline(x=best_average_size, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            plt.text(best_average_size + 1, 10, '$\langle \mathrm{TLD} \\rangle^{*} =' + f' {best_average_size}$', rotation=90)
            plt.savefig(plot_path)
        best_average_size = valid_average_sizes[np.argmin(mean_metric_scores)]
        print(f'Best average TAD size: {best_average_size}')
        return best_average_size

    @torch.no_grad()
    def predict(self, tad_out_dir, tad_filtered_dir, test_set, chrom_num, chrom_sizes, resolution, mean_tad_size, save_to_hdf5):
        bs = 1
        os.makedirs(tad_out_dir, exist_ok=False)
        os.makedirs(tad_filtered_dir, exist_ok=False)
        loader = DataLoader(test_set, bs, num_workers=0, pin_memory=False, )
        ctcf_df = pd.read_csv(self.ctcf_path, sep='\t', header=0, index_col=False)

        print('Predicting...')
        cell_pred_dfs = []
        filtered_dfs = []
        for idx, batch in enumerate(tqdm(loader)):
            z = batch.x.detach().cpu().numpy()
            num_clusters = z.shape[0] // mean_tad_size
            connectivity = create_linear_connectivity_matrix(z.shape[0])
            z = PCA(n_components=8, whiten=True).fit_transform(z) # current best 4
            # z = FastICA(n_components=4).fit_transform(z) # current best 4
            clusterer = AgglomerativeClustering(n_clusters=num_clusters, connectivity=connectivity, metric='euclidean', linkage='complete')
            pred = clusterer.fit_predict(z)

            current_ctcf_df = ctcf_df[ctcf_df['chrom'] == batch.chrom_name[0]]
            ctcf_vec = current_ctcf_df['pos_count'].values + current_ctcf_df['neg_count'].values

            pred = get_segments_with_higher_ctcf(z, pred, ctcf_vec)

            df = self.convert_batch_tad_preds_to_df(pred, batch.chrom_name[0], chrom_sizes, resolution)

            short_cell_name = batch.cell_name[0].split('/')[-1]
            if save_to_hdf5:
                h5_path = os.path.join(tad_out_dir, f'tad.h5')
                if idx % chrom_num != chrom_num - 1:
                    cell_pred_dfs.append(df)
                else:
                    cell_pred_dfs.append(df)
                    cell_pred_df = pd.concat(cell_pred_dfs).reset_index(drop=True)
                    cell_pred_df.to_hdf(h5_path, key=short_cell_name, mode='a')
                    cell_pred_dfs = []
            else:
                cell_csv_path = os.path.join(tad_out_dir, f'{short_cell_name}.csv')
                df.to_csv(
                    cell_csv_path, sep='\t', header=not os.path.exists(cell_csv_path),
                    index=False, mode='a', float_format='%.5f'
                )
            segments = convert_segmentation_to_tuples(df['score'].values, df['chrom'].values)
            chrom_list = []
            start_list = []
            end_list = []
            for i, (seg_start, seg_end) in enumerate(segments):
                chrom_list.append(df['chrom'].iloc[seg_start])
                start_list.append(df['start'].iloc[seg_start])
                end_list.append(df['end'].iloc[seg_end - 1])
            region_df = pd.DataFrame({'chrom': chrom_list, 'start': start_list, 'end': end_list})

            if save_to_hdf5:
                filtered_h5_path = os.path.join(tad_filtered_dir, f'tad.h5')
                if idx % chrom_num != chrom_num - 1:
                    filtered_dfs.append(region_df)
                else:
                    filtered_dfs.append(region_df)
                    filtered_df = pd.concat(filtered_dfs).reset_index(drop=True)
                    filtered_df.to_hdf(filtered_h5_path, key=short_cell_name, mode='a')
                    filtered_dfs = []
            else:
                region_csv_path = os.path.join(tad_filtered_dir, f'{short_cell_name}.csv')
                region_df.to_csv(region_csv_path, sep='\t', header=not os.path.exists(region_csv_path), index=False, mode='a')

        print('Done!')

    def convert_batch_tad_preds_to_df(self, pred, chrom_name, chrom_sizes, resolution):
        df = create_bin_df(chrom_sizes, resolution, [chrom_name])
        assert len(df) == len(pred)
        # Create new preds according to contiguous predicted segments
        new_pred = np.zeros_like(pred)
        prev_pred = pred[0]
        cluster_counter = 0
        for i in range(len(pred)):
            if pred[i] != prev_pred:
                cluster_counter += 1
                prev_pred = pred[i]
            new_pred[i] = cluster_counter
        assert len(new_pred) == len(df)
        df['score'] = new_pred
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


