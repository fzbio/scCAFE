import copy
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ReLU, Sigmoid, Conv1d

from torch_geometric.nn import LEConv
from torch_geometric.typing import SparseTensor, torch_sparse
from torch_geometric.utils import add_remaining_self_loops, scatter, softmax
from torch_geometric.data import Data, Batch
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')



# def torch_algrelmax(x: Tensor, width=31):
#     # https://discuss.pytorch.org/t/pytorch-argrelmax-or-c-function/36404
#     assert width % 2 == 1
#     peak_mask = torch.cat(
#         [x.new_zeros(1, dtype=torch.uint8), (x[:-2] < x[1:-1]) & (x[2:] < x[1:-1]), x.new_zeros(1, dtype=torch.uint8)],
#         dim=0)
#
#     b = torch.nn.functional.max_pool1d_with_indices(x.view(1, 1, -1), width, 1, padding=width // 2)[1].unique()
#     b = b[peak_mask[b].nonzero()]
#     return b


def torch_algrelmax(x: Tensor, width=31):
    # https://discuss.pytorch.org/t/pytorch-argrelmax-or-c-function/36404
    window_maxima = torch.nn.functional.max_pool1d_with_indices(x.view(1, 1, -1), width, 1, padding=width // 2)[
        1].squeeze()
    candidates = window_maxima.unique()
    nice_peaks = candidates[(window_maxima[candidates] == candidates).nonzero()]
    return nice_peaks


class GraphChannelSELayer1d(torch.nn.Module):
    """
    This code comes from https://github.com/ioanvl/1d_channel-spatial_se_layers
    Modified to fit the code style and the needs of this project.
    """
    def __init__(self, num_channels, reduction_ratio=4):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much the num_channels should be reduced
        """
        super(GraphChannelSELayer1d, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio
        self.fc1 = Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = Linear(num_channels_reduced, num_channels, bias=True)
        self.activ_1 = ReLU()
        self.activ_2 = Sigmoid()

    def forward(self, x):
        squeeze_tensor = x.view(1, -1, self.num_channels).mean(dim=1)
        fc_out_1 = self.activ_1(self.fc1(squeeze_tensor))
        fc_out_2 = self.activ_2(self.fc2(fc_out_1))
        output_x = x * fc_out_2
        return output_x.view(-1, self.num_channels)



class AttentionLEConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 GNN_topology: Optional[Callable] = None, GNN_seq: Optional[Callable] = None,
                 dropout: float = 0.0, negative_slope: float = 0.2, add_self_loops: bool = False,
                 SE_ratio = 4, topology_gnn_kw_dict: Optional[dict] = None, seq_gnn_kw_dict: Optional[dict] = None):
        super().__init__()

        self.in_channels = in_channels
        LE_out_channels = out_channels // 2
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN_topology = GNN_topology
        self.GNN_seq = GNN_seq
        self.add_self_loops = add_self_loops

        self.lin_topology = Linear(in_channels, in_channels)
        self.att_topology = Linear(2 * in_channels, 1)
        self.gnn_score_topology = LEConv(2 * self.in_channels, LE_out_channels)

        self.lin_seq = Linear(in_channels, in_channels)
        self.att_seq = Linear(2 * in_channels, 1)
        self.gnn_score_seq = LEConv(2 * self.in_channels, LE_out_channels)
        self.se = GraphChannelSELayer1d(2 * LE_out_channels, SE_ratio)
        self.final_lin = Linear(2 * LE_out_channels, 2 * LE_out_channels)

        if topology_gnn_kw_dict is None:
            topology_gnn_kw_dict = {}
        if seq_gnn_kw_dict is None:
            seq_gnn_kw_dict = {}

        if self.GNN_topology is not None:
            self.gnn_intra_cluster_topology = GNN_topology(self.in_channels, self.in_channels,
                                         **topology_gnn_kw_dict)
        else:
            self.gnn_intra_cluster_topology = None

        if self.GNN_seq is not None:
            self.gnn_intra_cluster_seq = GNN_seq(self.in_channels, self.in_channels,
                                         **seq_gnn_kw_dict)
        else:
            self.gnn_intra_cluster_seq = None
        # self.reset_parameters()

    # def reset_parameters(self):
    #     pass

    def compute_attention(self, x, edge_index, edge_weight, N, gnn_intra_cluster, lin, att):
        x_pool = x
        if gnn_intra_cluster is not None:
            if edge_weight is not None:
                x_pool = gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)
            else:
                x_pool = gnn_intra_cluster(x=x, edge_index=edge_index)

        x_pool_j = x_pool[edge_index[0]]
        x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce='max', dim_size=N)
        x_q = lin(x_q)[edge_index[1]]

        score = att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N)
        return score

    @staticmethod
    def create_seq_graph_edges(x, batch):
        num_nodes_per_example = scatter(
            torch.ones_like(batch), batch, dim=0, reduce='sum'
        )
        seq_data_list = []
        for num_nodes in num_nodes_per_example.tolist():
            edge_index_seq1 = torch.cat(
                [torch.arange(num_nodes - 1).view(1, -1), torch.arange(1, num_nodes).view(1, -1)], dim=0
            ).to(batch.device)
            edge_index_seq2 = torch.cat(
                [torch.arange(1, num_nodes).view(1, -1), torch.arange(num_nodes - 1).view(1, -1)], dim=0
            ).to(batch.device)
            edge_index_seq = torch.cat([edge_index_seq1, edge_index_seq2], dim=-1)
            seq_data_list.append(Data(edge_index=edge_index_seq, num_nodes=num_nodes))
        batch_obj_seq = Batch.from_data_list(seq_data_list)
        edge_index_seq = batch_obj_seq.edge_index
        return edge_index_seq

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:

        N = x.size(0)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # Create linearly connected graph edges.
        edge_index_seq = self.create_seq_graph_edges(x, batch)

        # Compute attention coefficients.
        score_topology = self.compute_attention(
            x, edge_index, edge_weight, N, self.gnn_intra_cluster_topology, self.lin_topology, self.att_topology
        )
        score_seq = self.compute_attention(
            x, edge_index_seq, None, N, self.gnn_intra_cluster_seq, self.lin_seq, self.att_seq
        )

        # Sample attention coefficients stochastically.
        score_topology = F.dropout(score_topology, p=self.dropout, training=self.training)
        score_seq = F.dropout(score_seq, p=self.dropout, training=self.training)

        v_topology_j = x[edge_index[0]] * score_topology.view(-1, 1)
        x_topology = scatter(v_topology_j, edge_index[1], dim=0, reduce='sum', dim_size=N)
        v_seq_j = x[edge_index_seq[0]] * score_seq.view(-1, 1)
        x_seq = scatter(v_seq_j, edge_index_seq[1], dim=0, reduce='sum', dim_size=N)
        x = torch.cat([x_topology, x_seq], dim=-1)

        # Cluster selection.
        fitness_topology = self.gnn_score_topology(x, edge_index)
        fitness_seq = self.gnn_score_seq(x, edge_index_seq)
        fitness = torch.cat([fitness_topology, fitness_seq], dim=-1)
        fitness = self.se(fitness).relu()
        fitness = self.final_lin(fitness)
        return fitness




class TADPooling(torch.nn.Module):
    r"""The Adaptive Structure Aware Pooling operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
    Graph Representations" <https://arxiv.org/abs/1911.07979>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`. (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            using intra-cluster properties.
            Especially helpful for graphs with higher degree of neighborhood
            (one of :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv` or
            any GNN which supports the :obj:`edge_weight` parameter).
            (default: :obj:`None`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        add_self_loops (bool, optional): If set to :obj:`True`, will add self
            loops to the new graph connectivity. (default: :obj:`False`)
        **kwargs (optional): Additional parameters for initializing the
            graph neural network layer.
    """
    def __init__(self, in_channels: int,
                 GNN_topology: Optional[Callable] = None, GNN_seq: Optional[Callable] = None,
                 dropout: float = 0.0, negative_slope: float = 0.2, add_self_loops: bool = False,
                 LE_out_channels: int = 8, SE_ratio = 4, smoothing_filter_size = 3,
                 local_maximum_window_size = 31, topology_gnn_kw_dict: Optional[dict] = None,
                 seq_gnn_kw_dict: Optional[dict] = None):
        super().__init__()

        self.in_channels = in_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN_topology = GNN_topology
        self.GNN_seq = GNN_seq
        self.add_self_loops = add_self_loops

        self.lin_topology = Linear(in_channels, in_channels)
        self.att_topology = Linear(2 * in_channels, 1)
        self.gnn_score_topology = LEConv(2 * self.in_channels, LE_out_channels)

        self.lin_seq = Linear(in_channels, in_channels)
        self.att_seq = Linear(2 * in_channels, 1)
        self.gnn_score_seq = LEConv(2 * self.in_channels, LE_out_channels)
        self.se = GraphChannelSELayer1d(2 * LE_out_channels, SE_ratio)
        self.smoothing_filter = Conv1d(1, 1, smoothing_filter_size, padding='same')
        self.local_maximum_window_size = local_maximum_window_size

        if topology_gnn_kw_dict is None:
            topology_gnn_kw_dict = {}
        if seq_gnn_kw_dict is None:
            seq_gnn_kw_dict = {}

        if self.GNN_topology is not None:
            self.gnn_intra_cluster_topology = GNN_topology(self.in_channels, self.in_channels,
                                         **topology_gnn_kw_dict)
        else:
            self.gnn_intra_cluster_topology = None

        if self.GNN_seq is not None:
            self.gnn_intra_cluster_seq = GNN_seq(self.in_channels, self.in_channels,
                                         **seq_gnn_kw_dict)
        else:
            self.gnn_intra_cluster_seq = None
        # self.reset_parameters()

    # def reset_parameters(self):
    #     r"""Resets all learnable parameters of the module."""
    #     self.lin.reset_parameters()
    #     self.att.reset_parameters()
    #     self.gnn_score.reset_parameters()
    #     if self.gnn_intra_cluster is not None:
    #         self.gnn_intra_cluster.reset_parameters()

    def compute_attention(self, x, edge_index, edge_weight, N, gnn_intra_cluster, lin, att):
        x_pool = x
        if gnn_intra_cluster is not None:
            x_pool = gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)

        x_pool_j = x_pool[edge_index[0]]
        x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce='max', dim_size=N)
        x_q = lin(x_q)[edge_index[1]]

        score = att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N)
        return score

    @staticmethod
    def create_seq_graph_edges(x, batch):
        num_nodes_per_example = scatter(
            torch.ones_like(batch), batch, dim=0, reduce='sum'
        )
        seq_data_list = []
        for num_nodes in num_nodes_per_example.tolist():
            edge_index_seq1 = torch.cat(
                [torch.arange(num_nodes - 1).view(1, -1), torch.arange(1, num_nodes).view(1, -1)], dim=0
            ).to(batch.device)
            edge_index_seq2 = torch.cat(
                [torch.arange(1, num_nodes).view(1, -1), torch.arange(num_nodes - 1).view(1, -1)], dim=0
            ).to(batch.device)
            edge_index_seq = torch.cat([edge_index_seq1, edge_index_seq2], dim=-1)
            seq_data_list.append(Data(edge_index=edge_index_seq, num_nodes=num_nodes))
        batch_obj_seq = Batch.from_data_list(seq_data_list)
        edge_index_seq = batch_obj_seq.edge_index
        return edge_index_seq

    def find_local_maximum(self, fitness_batch, batch):
        num_graph = torch.max(batch) + 1
        local_max_list = []
        num_nodes_per_example = scatter(
            torch.ones_like(batch), batch, dim=0, reduce='sum'
        )
        cum_num_nodes = torch.cat(
            [num_nodes_per_example.new_zeros(1),
             num_nodes_per_example.cumsum(dim=0)[:-1]], dim=0)
        for i in range(num_graph):
            fitness = fitness_batch[batch == i]
            width = self.local_maximum_window_size  # odd
            peak_mask = torch.cat([fitness.new_zeros(1, dtype=torch.uint8), (fitness[:-2] < fitness[1:-1]) & (fitness[2:] < fitness[1:-1]),
                                   fitness.new_zeros(1, dtype=torch.uint8)], dim=0)
            local_max = F.max_pool1d_with_indices(fitness.view(1, 1, -1), width, 1, padding=width // 2)[1].unique()
            local_max = local_max[peak_mask[local_max].nonzero().view(-1)]
            if local_max.size(0) == 0:
                local_max = torch.argmax(fitness.view(1, -1), dim=-1)
            local_max += cum_num_nodes[i]
            local_max_list.append(local_max)
        return torch.cat(local_max_list, dim=0)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:

        N = x.size(0)

        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value=1., num_nodes=N)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # Create linearly connected graph edges.
        edge_index_seq = self.create_seq_graph_edges(x, batch)

        # Compute attention coefficients.
        score_topology = self.compute_attention(
            x, edge_index, edge_weight, N, self.gnn_intra_cluster_topology, self.lin_topology, self.att_topology
        )
        score_seq = self.compute_attention(
            x, edge_index_seq, None, N, self.gnn_intra_cluster_seq, self.lin_seq, self.att_seq
        )

        # Sample attention coefficients stochastically.
        score_topology = F.dropout(score_topology, p=self.dropout, training=self.training)
        score_seq = F.dropout(score_seq, p=self.dropout, training=self.training)

        v_topology_j = x[edge_index[0]] * score_topology.view(-1, 1)
        x_topology = scatter(v_topology_j, edge_index[1], dim=0, reduce='sum', dim_size=N)
        v_seq_j = x[edge_index_seq[0]] * score_seq.view(-1, 1)
        x_seq = scatter(v_seq_j, edge_index_seq[1], dim=0, reduce='sum', dim_size=N)
        x = torch.cat([x_topology, x_seq], dim=-1)

        # Cluster selection.
        fitness_topology = self.gnn_score_topology(x, edge_index)
        fitness_seq = self.gnn_score_seq(x, edge_index_seq)
        fitness = torch.cat([fitness_topology, fitness_seq], dim=-1)
        fitness = self.smoothing_filter(self.se(fitness).sum(dim=-1).view(1, -1))   # SE layer already deals with the dimensionality of the input, but the smoothing filter does not.
        fitness = F.sigmoid(fitness.view(-1))


        perm = self.find_local_maximum(fitness, batch)
        x = x[perm] * fitness[perm].view(-1, 1)
        batch = batch[perm]

        # Graph coarsening. Only coarsen the topology graph. Do not coarsen the linearly connected graph.
        row, col = edge_index[0], edge_index[1]
        A = SparseTensor(row=row, col=col, value=edge_weight,
                         sparse_sizes=(N, N))
        S = SparseTensor(row=row, col=col, value=score_topology, sparse_sizes=(N, N))

        S = torch_sparse.index_select(S, 1, perm)
        A = torch_sparse.matmul(torch_sparse.matmul(torch_sparse.t(S), A), S)

        if self.add_self_loops:
            A = torch_sparse.fill_diag(A, 1.)
        else:
            A = torch_sparse.remove_diag(A)

        row, col, edge_weight = A.coo()
        edge_index = torch.stack([row, col], dim=0)

        return x, fitness, edge_index, edge_weight, batch, perm



if __name__ == '__main__':
    import torch_geometric.transforms as T
    from torch_geometric.datasets import TUDataset
    import os.path as osp
    from torch_geometric.nn import GCNConv
    from torch_geometric.loader import DataLoader
    # from torchviz import make_dot, make_dot_from_trace
    # from torchview import draw_graph
    # import graphviz

    path = osp.join(osp.dirname(osp.realpath(__file__)), 'test_data',
                    'PROTEINS')
    dataset = TUDataset(
        path,
        name='PROTEINS'
    )
    dataset = dataset.shuffle()
    n = (len(dataset) + 9) // 10
    test_dataset = dataset[:n]
    val_dataset = dataset[n:2 * n]
    train_dataset = dataset[2 * n:]
    test_loader = DataLoader(test_dataset, batch_size=20)
    val_loader = DataLoader(val_dataset, batch_size=20)
    train_loader = DataLoader(train_dataset, batch_size=2)

    data_batch = next(iter(train_loader)).to('cuda')
    # print(data_batch)

    net = TADPooling(
        dataset.num_features, GNN_topology=GCNConv, GNN_seq=GCNConv,
        LE_out_channels=8, SE_ratio=2, smoothing_filter_size=5, local_maximum_window_size=31,
        topology_gnn_kw_dict={}, seq_gnn_kw_dict={}
    ).to('cuda')
    # dot = make_dot(net(data_batch.x, data_batch.edge_index, None, data_batch.batch), params=dict(net.named_parameters()))
    # dot.format = 'png'
    # dot.render('test_data/model_architecture')

    # model_graph_1 = draw_graph(
    #     net, input_data=[data_batch.x, data_batch.edge_index, None, data_batch.batch],
    #     graph_name='TADPooling',
    #     # hide_inner_tensors=False,
    #     # hide_module_functions=False,
    # )
    # graph = model_graph_1.visual_graph
    # graph.format = 'png'
    # graph.render('test_data/model_architecture_2')

    print('original x', data_batch.x.shape)
    # print(data_batch.x)


    x, fitness, edge_index, edge_weight, batch, perm = net(data_batch.x, data_batch.edge_index, None, data_batch.batch)

    print('x', x.shape)
    print(x)
    print('fitness', fitness.shape)
    print('edge_index', edge_index.shape)
    print('edge_weight', edge_weight.shape)
    print('batch', batch.shape)
    print(batch)
    print('perm', perm.shape)
    print(perm)
