import cooler
import numpy as np
import pandas as pd
import sys
from copy import deepcopy
from scipy.sparse import coo_matrix
from scHiCTools.load.load_hic_file import get_chromosome_lengths
from scHiCTools.embedding import pairwise_distances, MDS, tSNE, PHATE, SpectralEmbedding, PCA
from scHiCTools.load.processing_utils import matrix_operation
from scHiCTools.analysis import kmeans, spectral_clustering, HAC
from tqdm.auto import tqdm


def load_HiC(file, genome_length, format=None, custom_format=None,
             header=0, chromosome=None, resolution=10000,
             resolution_adjust=True, map_filter=0., sparse=False, gzip=False,
             keep_n_strata=False, operations=None, sep=' ', **kwargs):
    """
    Load HiC contact map into a matrix
    Args:
        file (str): File path.
        genome_length (dict): The length of each genome.
        format (str): Now support .txt, .hic, and .mcool file.
        custom_format (int or list): If the format is not in our provided list.
        header (int): How many header lines to skip.
        chromosome (str): Specify the chromosome.
        resolution (int): Resolution.
        resolution_adjust (bool): In some situations, the input file is already pre-processed, and we don't need to adjust resolution again.
        map_filter (float): The threshold to filter some reads by map quality
        sparse (bool): Whether store in sparse matrices
        gzip (bool): Whether the file is zipped.
        keep_n_strata (int or None): Number of strata to keep.

    Return:
        Numpy.array: loaded contact map
    """
    size = genome_length[chromosome]
    mat = cooler.Cooler(file + '::' + f'/resolutions/{resolution}').matrix(balance=False, sparse=False).fetch(chromosome)


    if operations is not None:
        mat = matrix_operation(mat, operations, **kwargs)

    if keep_n_strata:
        strata = [np.diag(mat[i:, :len(mat) - i]) for i in range(keep_n_strata)]
    else:
        strata = None

    if sparse:
        mat = coo_matrix(mat)

    return mat, strata


class scHiCs(object):
    def __init__(self, list_of_files, reference_genome, resolution,
                 adjust_resolution=True, sparse=False, chromosomes='all',
                 format='customized', keep_n_strata=10, store_full_map=False,
                 operations=None, header=0, customized_format=None,
                 map_filter=0., gzip=False, sep=' ',
                 parallelize=False, n_processes=None, **kwargs):

        self.resolution = resolution
        self.chromosomes, self.chromosome_lengths = get_chromosome_lengths(reference_genome, chromosomes, resolution)
        self.num_of_cells = len(list_of_files)
        self.sparse = sparse
        self.keep_n_strata = keep_n_strata
        self.contacts = np.array([0] * len(list_of_files))
        self.short_range = np.array([0.0] * len(list_of_files))
        self.mitotic = np.array([0.0] * len(list_of_files))
        self.files = list_of_files
        self.strata = {
            ch: [np.zeros((self.num_of_cells, self.chromosome_lengths[ch] - i)) for i in range(keep_n_strata)]
            for ch in self.chromosomes} if keep_n_strata else None
        self.full_maps = None
        self.similarity_method = None
        self.distance = None

        assert keep_n_strata is not None or store_full_map is True

        if not store_full_map:
            self.full_maps = None
        elif sparse:
            self.full_maps = {ch: [None] * self.num_of_cells for ch in self.chromosomes}
        else:
            self.full_maps = {
                ch: np.zeros((self.num_of_cells, self.chromosome_lengths[ch], self.chromosome_lengths[ch]))
                for ch in self.chromosomes}

        print('Loading HiC data...')


        for idx, file in enumerate(tqdm(list_of_files)):
            # print('Processing {0} out of {1} files: {2}'.format(idx+1,len(list_of_files),file))

            for ch in self.chromosomes:
                if ('ch' in ch) and ('chr' not in ch):
                    ch = ch.replace("ch", "chr")
                mat, strata = load_HiC(
                    file, genome_length=self.chromosome_lengths,
                    format=format, custom_format=customized_format,
                    header=header, chromosome=ch, resolution=resolution,
                    resolution_adjust=adjust_resolution,
                    map_filter=map_filter, sparse=sparse, gzip=gzip,
                    keep_n_strata=keep_n_strata, operations=operations, sep=sep,
                    **kwargs)

                self.contacts[idx] += np.sum(mat) / 2 + np.trace(mat) / 2
                # ??
                self.short_range[idx] += sum(
                    [np.sum(mat[i, i:i + int(2000000 / self.resolution)]) for i in range(len(mat))])
                self.mitotic[idx] += sum(
                    [np.sum(mat[i, i + int(2000000 / self.resolution):i + int(12000000 / self.resolution)]) for i in
                     range(len(mat))])

                if store_full_map:
                    self.full_maps[ch][idx] = mat

                if keep_n_strata:
                    # self.strata[ch][idx] = strata
                    for strata_idx, stratum in enumerate(strata):
                        self.strata[ch][strata_idx][idx, :] = stratum

    def cal_strata(self, n_strata):
        """

        Alter the number of strata kept in a `scHiCs` object.


        Parameters
        ----------
        n_strata : int
            Number of strata to keep.

        Returns
        -------
        dict
            Strata of cells.

        """

        if self.full_maps is None:
            if self.keep_n_strata <= n_strata:
                print(' Only {0} strata are kept!'.format(self.keep_n_strata))
                return deepcopy(self.strata)
            else:
                return deepcopy({ch: self.strata[ch][:n_strata] for ch in self.chromosomes})
        else:
            if self.keep_n_strata is None:
                new_strata = {
                    ch: [np.zeros((self.num_of_cells, self.chromosome_lengths[ch] - i))
                         for i in range(n_strata)] for ch in self.chromosomes}
                for ch in self.chromosomes:
                    for idx in range(self.num_of_cells):
                        fmap = self.full_maps[ch][idx].toarray() if self.sparse else self.full_maps[ch][idx]
                        for i in range(n_strata):
                            new_strata[ch][i][idx, :] = np.diag(fmap[i:, :-i])
                return new_strata
            elif self.keep_n_strata >= n_strata:
                return deepcopy({ch: self.strata[ch][:n_strata] for ch in self.chromosomes})
            else:
                for ch in self.chromosomes:
                    self.strata[ch] += [(np.zeros(self.num_of_cells, self.chromosome_lengths[ch] - i))
                                        for i in range(self.keep_n_strata, n_strata)]
                    for idx in range(self.num_of_cells):
                        fmap = self.full_maps[ch][idx].toarray() if self.sparse else self.full_maps[ch][idx]
                        for i in range(self.keep_n_strata, n_strata):
                            self.strata[ch][i][idx, :] = np.diag(fmap[i:, :-i])
                return deepcopy(self.strata)

    def processing(self, operations, **kwargs):
        """

        Apply a smoothing method to contact maps.
        Requre the `scHiCs` object to store the full map of contacts maps.


        Parameters
        ----------
        operations : str
            The methods use for smoothing the maps.
            Avaliable operations: 'convolution', 'random_walk', 'network_enhancing'.

        **kwargs :
            Other arguments specify smoothing methods passed to function.
            See function `scHiCTools.load.processing_utils.matrix_operation`.


        Returns
        -------
        None.

        """

        if self.full_maps is None:
            raise ValueError('No full maps stored. Processing is not doable.')
        if self.sparse:
            for ch in self.chromosomes:
                for i, mat in enumerate(self.full_maps[ch]):
                    self.full_maps[ch][i] = coo_matrix(matrix_operation(mat.toarray(), operations, **kwargs))
        else:
            for ch in self.chromosomes:
                for i, mat in enumerate(self.full_maps[ch]):
                    self.full_maps[ch][i, :, :] = matrix_operation(mat, operations, **kwargs)
        # Update the strata
        if self.keep_n_strata is not None:
            for ch in self.chromosomes:
                for i, mat in enumerate(self.full_maps[ch]):
                    for j in range(self.keep_n_strata):
                        self.strata[ch][j][i, :] = np.diag(mat[j:, :len(mat) - j])


    def scHiCluster(self, dim=2, n_clusters=4, cutoff=0.8, n_PCs=10, **kwargs):

        """

        Embedding and clustering single cells using HiCluster.
        Reference:
        Zhou J, Ma J, Chen Y, Cheng C, Bao B, Peng J, et al.
        Robust single-cell Hi-C clustering by convolution- and random-walk–based imputation.
        PNAS. 2019 Jul 9;116(28):14011–8.


        Parameters
        ----------
        dim : int, optional
            Number of dimension of embedding. The default is 2.

        n_clusters : int, optional
            Number of clusters. The default is 4.

        cutoff : float, optional
            The cutoff proportion to convert the real contact
            matrix into binary matrix. The default is 0.8.

        n_PCs : int, optional
            Number of principal components. The default is 10.

        **kwargs :
            Other arguments passed to kmeans.
            See `scHiCTools.analysis.clustering.kmeans` function.

        Returns
        -------
        embeddings : numpy.ndarray
            The embedding of cells using HiCluster.

        label : numpy.ndarray
            An array of cell labels clustered by HiCluster.

        """

        if self.full_maps is None:
            raise ValueError('No full maps stored. scHiCluster is not doable.')

        X = None
        for ch in self.chromosomes:
            sys.stdout.write('\r')
            sys.stdout.write('HiCluster processing chromosome {}. '.format(ch))
            # print('HiCluster processing chromosomes {}'.format(ch))
            A = self.full_maps[ch].copy()
            if len(A.shape) == 3:
                n = A.shape[1] * A.shape[2]
                A.shape = (A.shape[0], n)
            A = np.quantile(A, cutoff, axis=1) < np.transpose(A)
            A = PCA(A.T, n_PCs)
            if X is None:
                X = A
            else:
                X = np.append(X, A, axis=1)

        X = PCA(X, n_PCs)
        label = kmeans(X, n_clusters, kwargs.pop('weights', None), kwargs.pop('iteration', 1000))

        return X[:, :dim], label

    def learn_embedding(self, similarity_method, embedding_method,
                        dim=2, aggregation='median', n_strata=None, return_distance=False,
                        print_time=False, parallelize=False, n_processes=1,
                        **kwargs):
        """

        Function to find a low-dimensional embedding for cells.


        Parameters
        ----------
        similarity_method : str
            The method used to calculate similarity matrix.
            Now support 'inner_product', 'HiCRep' and 'Selfish'.

        embedding_method : str
            The method used to project cells into lower-dimensional space.
            Now support 'MDS', 'tSNE', 'phate', 'spectral_embedding'.

        dim : int, optional
            Dimension of the embedding space.
            The default is 2.

        aggregation : str, optional
            Method to find the distance matrix based on distance matrices of chromesomes.
            Must be 'mean' or 'median'.
            The default is 'median'.

        n_strata : int, optional
            Number of strata used in calculation.
            The default is None.

        return_distance : bool, optional
            Whether to return the distance matrix of cells.
            If True, return (embeddings, distance_matrix);
            if False, only return embeddings.
            The default is False.

        print_time : bool, optional
            Whether to print process time. The default is False.

        **kwargs :
            Including two arguments for Selfish
            (see funciton `pairwise_distances`):\
            `n_windows`: number of Selfish windows\
            `sigma`: sigma in the Gaussian-like kernel\
            and some arguments specify different embedding method
            (see functions in `scHiCTools.embedding.embedding`).


        Returns
        -------
        embeddings: numpy.ndarray
            The embedding of cells in lower-dimensional space.

        final_distance: numpy.ndarray, optional
            The pairwise distance calculated.

        """

        if self.distance is None or self.similarity_method != similarity_method:
            self.similarity_method = similarity_method
            distance_matrices = []
            assert embedding_method.lower() in ['mds', 'tsne', 'umap', 'phate', 'spectral_embedding']
            assert n_strata is not None or self.keep_n_strata is not None
            n_strata = n_strata if n_strata is not None else self.keep_n_strata
            new_strata = self.cal_strata(n_strata)
            if print_time:
                time1 = 0
                time2 = 0
                for ch in self.chromosomes:
                    print(ch)
                    distance_mat, t1, t2 = pairwise_distances(new_strata[ch], similarity_method, print_time,
                                                              kwargs.get('sigma', .5), kwargs.get('window_size', 10),
                                                              parallelize, n_processes)
                    time1 = time1 + t1
                    time2 = time2 + t2
                    distance_matrices.append(distance_mat)
                print('Sum of time 1:', time1)
                print('Sum of time 2:', time2)
            else:
                for i, ch in enumerate(self.chromosomes):
                    # print(ch)
                    distance_mat = pairwise_distances(new_strata[ch],
                                                      similarity_method,
                                                      print_time,
                                                      kwargs.get('sigma', .5),
                                                      kwargs.get('window_size', 10))
                    distance_matrices.append(distance_mat)
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-30s] %d/%d \t Calculating chromosome %s. " % (
                    '=' * int((i + 1) / len(self.chromosomes) * 30), i + 1, len(self.chromosomes), ch))

            self.distance = np.array(distance_matrices)

        if aggregation == 'mean':
            final_distance = np.mean(self.distance, axis=0)
        elif aggregation == 'median':
            final_distance = np.median(self.distance, axis=0)
        else:
            raise ValueError('Aggregation method {0} not supported. Only "mean" or "median".'.format(aggregation))

        np.fill_diagonal(final_distance, 0)

        embedding_method = embedding_method.lower()
        if embedding_method == 'mds':
            embeddings = MDS(final_distance, dim)
        elif embedding_method == 'tsne':
            embeddings = tSNE(final_distance, dim,
                              kwargs.pop('perp', 30),
                              kwargs.pop('iteration', 1000),
                              kwargs.pop('momentum', 0.5),
                              kwargs.pop('rate', 200),
                              kwargs.pop('tol', 1e-5))
        # elif embedding_method == 'umap':
        #     embeddings = UMAP(final_distance, dim,
        #                       kwargs.pop('n',5),
        #                       kwargs.pop('min_dist',1),
        #                       kwargs.pop('n_epochs',10),
        #                       kwargs.pop('alpha',1),
        #                       kwargs.pop('n_neg_samples',0))
        elif embedding_method == 'phate':
            embeddings = PHATE(final_distance, dim,
                               kwargs.pop('k', 5),
                               kwargs.pop('a', 1),
                               kwargs.pop('gamma', 1),
                               kwargs.pop('t_max', 100),
                               kwargs.pop('momentum', .1),
                               kwargs.pop('iteration', 1000))
        elif embedding_method == 'spectral_embedding':
            graph = np.exp(-np.square(final_distance) / np.mean(final_distance ** 2))
            graph = graph - np.diag(graph.diagonal())
            embeddings = SpectralEmbedding(graph, dim)
        else:
            raise ValueError('Embedding method {0} not supported. '.format(embedding_method))

        if return_distance:
            return embeddings, final_distance
        else:
            return embeddings

    def clustering(self,
                   n_clusters,
                   clustering_method,
                   similarity_method,
                   aggregation='median',
                   n_strata=None,
                   print_time=False,
                   **kwargs):
        """

        Parameters
        ----------
        n_clusters : int
            Number of clusters.

        clustering_method : str
            Clustering method in 'kmeans', 'spectral_clustering' or 'HAC'(hierarchical agglomerative clustering).

        similarity_method : str
            Reproducibility measure.
            Value in ‘InnerProduct’, ‘HiCRep’ or ‘Selfish’.

        aggregation : str, optional
             Method to aggregate different chromosomes.
             Value is either 'mean' or 'median'.
             The default is 'median'.

        n_strata : int or None, optional
            Only consider contacts within this genomic distance.
            If it is None, it will use the all strata kept from previous loading process.
            The default is None.

        print_time : bool, optional
            Whether to print the processing time. The default is False.

        **kwargs :
            Other arguments pass to function `scHiCTools.embedding.reproducibility.pairwise_distances `,
            and the clustering function in `scHiCTools.analysis.clustering`.


        Returns
        -------
        label : numpy.ndarray
            An array of cell labels clustered.

        """
        if self.distance is None or self.similarity_method != similarity_method:
            self.similarity_method = similarity_method
            distance_matrices = []
            assert n_strata is not None or self.keep_n_strata is not None
            n_strata = n_strata if n_strata is not None else self.keep_n_strata
            new_strata = self.cal_strata(n_strata)

            for i, ch in enumerate(self.chromosomes):
                # print(ch)
                distance_mat = pairwise_distances(new_strata[ch],
                                                  similarity_method,
                                                  print_time,
                                                  kwargs.get('sigma', .5),
                                                  kwargs.get('window_size', 10))
                distance_matrices.append(distance_mat)
                sys.stdout.write('\r')
                sys.stdout.write("[%-30s] %d/%d \t Calculating chromosome %s. " % (
                '=' * int((i + 1) / len(self.chromosomes) * 30), i + 1, len(self.chromosomes), ch))
            self.distance = np.array(distance_matrices)

        if aggregation == 'mean':
            final_distance = np.mean(self.distance, axis=0)
        elif aggregation == 'median':
            final_distance = np.median(self.distance, axis=0)
        else:
            raise ValueError('Aggregation method {0} not supported. Only "mean" or "median".'.format(aggregation))

        np.fill_diagonal(final_distance, 0)

        clustering_method = clustering_method.lower()
        if clustering_method == 'kmeans':
            embeddings = MDS(final_distance, n_clusters)
            label = kmeans(embeddings,
                           k=n_clusters,
                           **kwargs)
        elif clustering_method == 'spectral_clustering':
            label = spectral_clustering(final_distance,
                                        data_type='distance_matrix',
                                        n_clusters=n_clusters,
                                        **kwargs)
        elif clustering_method == 'hac':
            label = HAC(final_distance,
                        'distance_matrix',
                        n_clusters,
                        kwargs.pop('method', 'centroid'))
        else:
            raise ValueError('Embedding method {0} not supported. '.format(clustering_method))

        return label
