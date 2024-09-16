from multiscale_calling import CrossStitchFeatureCaller
from dcn.dcn import DCN
import torch
from nn_data import get_hmm_split_dataset
from torch_geometric.loader import DataLoader
import numpy as np
from hmmlearn.hmm import GaussianHMM, GMMHMM
from middleware import MiddleWareDataset
from nn_data import get_cv_split_scools
from hmmlearn.hmm import GaussianHMM, GMMHMM
from schickit.utils import get_cg_content_bin_df
from scipy.stats import mannwhitneyu
from nn_data import expand_cell_ids_to_graph_ids, check_vec_unique
import fanc
from tqdm.auto import tqdm
from schickit.utils import create_bin_df, get_chrom_sizes
import os
import pandas as pd
from sklearn.decomposition import PCA


def average_every_n_rows(matrix, n):

    # Define the number of rows to take the mean over
    rows_per_group = n

    # Calculate the number of full groups and the number of remaining rows
    num_full_groups = matrix.size(0) // rows_per_group
    num_remaining_rows = matrix.size(0) % rows_per_group

    # Reshape the matrix to group full groups of rows
    full_groups_matrix = matrix[:num_full_groups * rows_per_group].view(-1, rows_per_group, matrix.size(1))

    # Compute the mean over the full groups
    mean_full_groups = full_groups_matrix.mean(dim=1)

    # If there are remaining rows, compute their mean separately
    if num_remaining_rows > 0:
        remaining_rows_matrix = matrix[num_full_groups * rows_per_group:]
        mean_remaining_rows = remaining_rows_matrix.mean(dim=0, keepdim=True)
        # Concatenate the means of full groups and remaining rows
        mean_matrix = torch.cat((mean_full_groups, mean_remaining_rows), dim=0)
    else:
        mean_matrix = mean_full_groups
    return mean_matrix



class CompartmentCaller(object):
    def __init__(self):
        pass

    def get_embeddings(self, loader):
        embeddings = []
        for batch in loader:
            z = batch.x
            embeddings.append(z.detach().cpu().numpy())
        return embeddings


    def flip_by_gc(self, proba_mat, cluster, chrom_bin_gc_content):
        proba_vec = proba_mat[np.arange(cluster.shape[0]), cluster]


        bin_gc_df = chrom_bin_gc_content

        gc_content_vecs = [bin_gc_df.iloc[np.where(cluster == c)]['gc'] for c in range(len(np.unique(cluster)))]
        gc_content_of_each_cluster = [vec.mean() for vec in gc_content_vecs]

        # Get the largest gc content index
        largest_gc_index = np.argmax(np.array(gc_content_of_each_cluster))
        # Compare the largest gc content cluster with the other clusters
        for i in range(len(gc_content_of_each_cluster)):
            if i != largest_gc_index:
                _, p = mannwhitneyu(gc_content_vecs[largest_gc_index], gc_content_vecs[i])
                # print(f'p-value between {largest_gc_index} and {i}: {p}')

        # print(f'AIC: {aic}; BIC: {bic}')
        # Flip the probability to negative if the cluster is not the largest gc content cluster
        for i in range(len(proba_vec)):
            if cluster[i] != largest_gc_index:
                proba_vec[i] = -proba_vec[i]
        return proba_vec


    @torch.no_grad()
    def predict_compartments(self, test_set: MiddleWareDataset, assembly_path, chrom_sizes_path, resolution, output_dir):
        os.makedirs(output_dir, exist_ok=False)
        dataset_for_training_HMM = test_set[:100]
        loader_for_training_HMM = DataLoader(
            dataset_for_training_HMM, 1, num_workers=0, pin_memory=False
        )
        training_preds = self.get_embeddings(loader_for_training_HMM)
        training_pred = np.concatenate(training_preds, axis=0)
        training_lengths = [len(p) for p in training_preds]
        hmm = GMMHMM(n_components=2, n_iter=100, covariance_type='diag')
        hmm.fit(training_pred, lengths=training_lengths)

        chrom_sizes = get_chrom_sizes(chrom_sizes_path)

        gc_content_dict = {chrom: get_cg_content_bin_df(assembly_path, [chrom], resolution, chrom_sizes) for chrom in chrom_sizes}

        test_loader = DataLoader(test_set, 1, num_workers=0, pin_memory=False)
        for ind, data in enumerate(tqdm(test_loader)):
            x = data.x.detach().cpu().numpy()

            proba = hmm.predict_proba(x)
            cluster = np.argmax(proba, axis=1)
            chrom_name = data.chrom_name[0]

            proba_vec = self.flip_by_gc(proba, cluster, gc_content_dict[chrom_name])

            df = self.convert_batch_compartment_preds_to_df(proba_vec, data.chrom_name[0], chrom_sizes, resolution)
            short_cell_name = data.cell_name[0].split('/')[-1]
            cell_csv_path = os.path.join(output_dir, f'{short_cell_name}.csv')
            df.to_csv(
                cell_csv_path, sep='\t', header=not os.path.exists(cell_csv_path),
                index=False, mode='a', float_format='%.5f'
            )

    def convert_batch_compartment_preds_to_df(self, preds, chrom_name, chrom_sizes, resolution):
        df = create_bin_df(chrom_sizes, resolution, [chrom_name])
        assert len(df) == len(preds)
        df['score'] = preds
        df = df.reset_index(drop=True)
        return df


    # @torch.no_grad()
    # def consensus_compartments(self, test_set: MiddleWareDataset, valid_chroms, assembly_path, chrom_sizes_path, resolution, output_path, parser_func):
    #     chrom_consensus_embeddings = {}
    #     test_loader = DataLoader(test_set, 1, num_workers=0, pin_memory=False)
    #
    #     valid_data_count = 0
    #     for ind, data in enumerate(tqdm(test_loader)):
    #         if parser_func(data.cell_name[0]):
    #             valid_data_count += 1
    #             x = data.x.detach().cpu().numpy()
    #             chrom_name = data.chrom_name[0]
    #             if chrom_name not in chrom_consensus_embeddings:
    #                 chrom_consensus_embeddings[chrom_name] = x
    #             else:
    #                 chrom_consensus_embeddings[chrom_name] = chrom_consensus_embeddings[chrom_name] + x
    #     assert valid_data_count == len(test_set)
    #     assert valid_data_count % len(valid_chroms) == 0
    #     cell_num = valid_data_count // len(valid_chroms)
    #     for chrom_name in chrom_consensus_embeddings:
    #         chrom_consensus_embeddings[chrom_name] = chrom_consensus_embeddings[chrom_name] / cell_num
    #     chrom_sizes = get_chrom_sizes(chrom_sizes_path)
    #     gc_content_dict = {chrom: get_cg_content_bin_df(assembly_path, [chrom], resolution, chrom_sizes) for chrom in valid_chroms}
    #     all_x = np.concatenate([chrom_consensus_embeddings[chrom_name] for chrom_name in valid_chroms], axis=0)
    #     all_lengths = [len(chrom_consensus_embeddings[chrom_name]) for chrom_name in valid_chroms]
    #     hmm = GMMHMM(n_components=2, n_iter=100, covariance_type='diag')
    #     hmm.fit(all_x, lengths=all_lengths)
    #
    #     for chrom_name in valid_chroms:
    #         x = chrom_consensus_embeddings[chrom_name]
    #         proba = hmm.predict_proba(x)
    #         cluster = np.argmax(proba, axis=1)
    #         proba_vec = self.flip_by_gc(proba, cluster, gc_content_dict[chrom_name])
    #         df = self.convert_batch_compartment_preds_to_df(proba_vec, chrom_name, chrom_sizes, resolution)
    #         df.to_csv(
    #             output_path, sep='\t', header=not os.path.exists(output_path),
    #             index=False, mode='a', float_format='%.5f'
    #         )


    @torch.no_grad()
    def consensus_compartments(self, pred_paths, output_path):
        pred_dfs = [pd.read_csv(path, sep='\t', header=0, index_col=False) for path in pred_paths]
        mean_vec = np.mean(np.array([df['score'].values for df in pred_dfs]), axis=0)
        consensus_df = pred_dfs[0].copy()
        consensus_df['score'] = mean_vec
        consensus_df.to_csv(output_path, sep='\t', header=True, index=False, float_format='%.5f')

