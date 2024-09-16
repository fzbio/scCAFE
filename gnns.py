import torch
from torch_geometric.nn import SAGEConv, BatchNorm, LayerNorm, VGAE, GAE
from torch import Tensor
from torch.nn import Conv1d
from layers import AttentionLEConv
from nn_data import short_dist_neg_sampling
from torch_geometric.nn.inits import reset
from cross_stitch import CrossStitchUnit
from typing import Union
from torch.nn import MultiheadAttention

EPS = 1e-15


class CrossStitchedGAE(torch.nn.Module):
    def __init__(self, tasks: list, backbone: torch.nn.ModuleDict, heads: torch.nn.ModuleDict,
                 encoder_stages: Union[list, None], decoder_stages: list,
                 encoder_channels: dict, decoder_channels: dict, alpha: float, beta: float,
                 freeze_cross_stitch: bool = False
                 ):
        super(CrossStitchedGAE, self).__init__()

        # Tasks, backbone and heads
        self.tasks = tasks
        self.backbone = backbone
        self.heads = heads
        self.encoder_stages = encoder_stages
        self.decoder_stages = decoder_stages

        # Cross-stitch units
        if self.encoder_stages is not None:
            self.cross_stitch_encoder = torch.nn.ModuleDict(
                {stage: CrossStitchUnit(self.tasks, encoder_channels[stage], alpha, beta) for stage in self.encoder_stages})
        else:
            self.cross_stitch_encoder = torch.nn.ModuleDict({})
        self.cross_stitch_decoder = torch.nn.ModuleDict(
            {stage: CrossStitchUnit(self.tasks, decoder_channels[stage], alpha, beta) for stage in self.decoder_stages})
        if freeze_cross_stitch:
            self.freeze_cross_stitch_units(self.cross_stitch_encoder)
            self.freeze_cross_stitch_units(self.cross_stitch_decoder)

    def freeze_cross_stitch_units(self, unit: torch.nn.ModuleDict):
        for stage in unit:
            stage_unit = unit[stage]
            for param in stage_unit.parameters():
                param.requires_grad = False

    def encode(self, x, edge_index):
        x = {task: x for task in self.tasks}
        edge_index = {task: edge_index for task in self.tasks}

        if self.encoder_stages is None:
            x = {task: self.backbone[task].encode_stage(x[task], edge_index[task], '') for task in self.tasks}
            return x

        else:
            # Backbone
            for stage in self.encoder_stages:
                for task in self.tasks:
                    x[task] = self.backbone[task].encode_stage(x[task], edge_index[task], stage)

                # Cross-stitch the task-specific features
                x = self.cross_stitch_encoder[stage](x)
            return x

    def decode(self, z: dict, edge_index: dict):
        # Backbone
        z = self.compute_backbone_output(z)

        # Task-specific heads
        out = self.compute_head_from_backbone_output(z, edge_index)
        return out

    def compute_backbone_output(self, z: dict):
        for stage in self.decoder_stages:

            # Forward through next stage of task-specific network
            for task in self.tasks:
                z[task] = self.backbone[task].decode_stage(z[task], stage)

            # Cross-stitch the task-specific features
            z = self.cross_stitch_decoder[stage](z)
        return z

    def compute_head_from_backbone_output(self, z: dict, edge_index: dict):
        out = {}
        for task in z.keys():
            if 'loop' in task:
                out[task] = self.heads[task](z[task], edge_index[task], sigmoid=True)
            else:
                out[task] = self.heads[task](z[task], sigmoid=False)
        return out

    def get_losses(self, backbone_output_z: dict, data, neg_edge_index=None, ls=0.1):
        losses = {}
        for task in self.tasks:
            if 'loop' in task:
                pos_pred = self.heads[task](backbone_output_z[task], data.edge_label_index)
                if neg_edge_index is None:
                    neg_edge_index = short_dist_neg_sampling(data.edge_label_index, data.edge_index, data.num_nodes)
                neg_pred = self.heads[task](backbone_output_z[task], neg_edge_index)
                # pos_loss = -torch.log(pos_pred + EPS).mean()
                # neg_loss = -torch.log(1 - neg_pred + EPS).mean()
                pos_loss = (- (1 - ls) * torch.log(pos_pred + EPS) - ls * torch.log(1 - pos_pred + EPS)).mean()
                neg_loss = (- (1 - ls) * torch.log(1 - neg_pred + EPS) - ls * torch.log(neg_pred + EPS)).mean()
                losses[task] = pos_loss + neg_loss
            elif task == 'node_reg':
                mse = torch.nn.MSELoss(reduction='mean')
                preds = self.heads[task](backbone_output_z[task])
                losses[task] = mse(preds, data.tad_label)
            else:
                raise NotImplementedError
        return losses

    @torch.no_grad()
    def get_evaltime_loss_loop_pred(self, pos_pred, neg_pred, ls=0.1):
        # pos_loss = -torch.log(pos_pred + EPS).mean()
        # neg_loss = -torch.log(1 - neg_pred + EPS).mean()
        pos_loss = (- (1 - ls) * torch.log(pos_pred + EPS) - ls * torch.log(1 - pos_pred + EPS)).mean()
        neg_loss = (- (1 - ls) * torch.log(1 - neg_pred + EPS) - ls * torch.log(neg_pred + EPS)).mean()
        return pos_loss + neg_loss

    @torch.no_grad()
    def get_evaltime_loss_node_reg(self, pred, label):
        mse = torch.nn.MSELoss(reduction='mean')
        return mse(pred, label)


class CrossStitchedVGAE(CrossStitchedGAE):
    def __init__(self, tasks: list, backbone: torch.nn.ModuleDict, heads: torch.nn.ModuleDict,
                 encoder_stages: Union[list, None], decoder_stages: list,
                 encoder_channels: dict, decoder_channels: dict, alpha: float, beta: float,
                 freeze_cross_stitch: bool = False):
        super().__init__(tasks, backbone, heads,
                 encoder_stages, decoder_stages,
                 encoder_channels, decoder_channels, alpha, beta,
                 freeze_cross_stitch)

    def get_kl_loss_of_task(self, task):
        return self.backbone[task].kl_loss()


class SingleTaskGAE(GAE):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def encode_stage(self, x, edge_index, stage):
        return self.encode(x, edge_index)

    def decode_stage(self, z, stage):
        assert stage in ['decoder_layer1']

        if stage == 'decoder_layer1':
            z = self.decoder.decoder_layer1(z)
            return z

        else:
            layer = getattr(self.decoder, stage)
            return layer(z)


class SingleTaskVGAE(VGAE):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def encode_stage(self, x, edge_index, stage):
        return self.encode(x, edge_index)

    def decode_stage(self, z, stage):
        assert stage in ['decoder_layer1']

        if stage == 'decoder_layer1':
            z = self.decoder.decoder_layer1(z)
            return z

        else:
            layer = getattr(self.decoder, stage)
            return layer(z)


class DenseSingleTaskDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.decoder_layer1 = torch.nn.Sequential(
            # torch.nn.ReLU(),
            # Permute((1, 0)),
            # Conv1d(input_dim, 512, 3, padding='same'),
            # Conv1d(512, 512, 3, padding='same'),
            # Permute((1, 0)),
            # BatchNorm(512, momentum=0.01),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.2),
            # torch.nn.Linear(input_dim, output_dim),
            # BatchNorm(256, momentum=0.01),
            # torch.nn.ReLU(),
        )

        # self.decoder_layer2 = torch.nn.Sequential(
        #     # torch.nn.Dropout(p=0.2),
        #     # Permute((1, 0)),
        #     # Conv1d(256, 256, 3, padding='same'),
        #     # Permute((1, 0)),
        #     # # BatchNorm(256, momentum=0.01),
        #     # torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.2),
        #     torch.nn.Linear(256, 128),
        #     # BatchNorm(128, momentum=0.01),
        #     torch.nn.ReLU(),
        # )

        # self.decoder_layer3 = torch.nn.Sequential(
        #     # torch.nn.Dropout(p=0.2),
        #     # Permute((1, 0)),
        #     # Conv1d(128, 128, 3, padding='same'),
        #     # Permute((1, 0)),
        #     # torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.2),
        #     torch.nn.Linear(128, output_dim),
        #     # BatchNorm(64, momentum=0.01),
        #     torch.nn.ReLU(),
        # )

    def forward(self, z, sigmoid=False):
        z = self.decoder_layer1(z)
        # z = self.decoder_layer2(z)
        # z = self.decoder_layer3(z)
        return z


class EdgeHead(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding_layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
        )
        self.compressing_layers = torch.nn.Sequential(
            # BatchNorm(16, momentum=0.01),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, z, edge_index, sigmoid=True):
        edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        z = self.embedding_layers(edge_features)
        z = self.compressing_layers(z).view(-1)
        return torch.sigmoid(z) if sigmoid else z


class NodeHead(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding_layers = torch.nn.Sequential(
            Permute((1, 0)),
            Conv1d(input_dim, 96, 3, padding='same'),
            torch.nn.ReLU(),
            Conv1d(96, 96, 3, padding='same'),
            torch.nn.ReLU(),
            Permute((1, 0)),
            torch.nn.Linear(96, 64),
            # BatchNorm(32, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(64, 32),
            # BatchNorm(32, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(32, 16),
        )
        self.compressing_layers = torch.nn.Sequential(
            # BatchNorm(16, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(16, 1)
        )

    def forward(self, z, sigmoid=False):
        z = self.embedding_layers(z)
        z = self.compressing_layers(z).view(-1)
        return torch.sigmoid(z) if sigmoid else z


class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    Implemented by @Mikoto10032 on GitHub
    https://github.com/Mikoto10032/AutomaticWeightedLoss/tree/master
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class MultiTaskVGAE(VGAE):
    def __init__(self, encoder, edge_decoder, recon_decoder):
        super().__init__(encoder, edge_decoder)
        # self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.recon_decoder = recon_decoder
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.edge_decoder)
        reset(self.recon_decoder)

    def decode_recon(self, z, edge_index):
        return self.recon_decoder(z, edge_index)

    def decode_edge(self, z, edge_index):
        return self.edge_decoder(z, edge_index)



class AttentionLeEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = AttentionLEConv(in_channels, 2 * out_channels, SAGEConv, SAGEConv,
                topology_gnn_kw_dict={'aggr': 'mean'}, seq_gnn_kw_dict={'aggr': 'mean'})
        self.conv2 = AttentionLEConv(2 * out_channels, out_channels, SAGEConv, SAGEConv,
                topology_gnn_kw_dict={'aggr': 'mean'}, seq_gnn_kw_dict={'aggr': 'mean'})

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalAttentionLeEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = AttentionLEConv(in_channels, 2 * out_channels, SAGEConv, SAGEConv,
                topology_gnn_kw_dict={'aggr': 'mean'}, seq_gnn_kw_dict={'aggr': 'mean'})
        self.conv_mu = AttentionLEConv(2 * out_channels, out_channels, SAGEConv, SAGEConv,
                topology_gnn_kw_dict={'aggr': 'mean'}, seq_gnn_kw_dict={'aggr': 'mean'})
        self.conv_logstd = AttentionLEConv(2 * out_channels, out_channels, SAGEConv, SAGEConv,
                topology_gnn_kw_dict={'aggr': 'mean'}, seq_gnn_kw_dict={'aggr': 'mean'})

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class GraphSageEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels, aggr='mean')
        self.conv2 = SAGEConv(2 * out_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# class GraphSageEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = SAGEConv(in_channels, out_channels, aggr='mean')
#
#     def forward(self, x, edge_index):
#         return self.conv1(x, edge_index)



class VariationalGraphSageEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels, aggr='mean')
        self.conv_mu = SAGEConv(2 * out_channels, out_channels, aggr='mean')
        self.conv_logstd = SAGEConv(2 * out_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class EdgeMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding_layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 96),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(96, 64),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
        )
        self.compressing_layers = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        z = self.embedding_layers(x)
        y_logit = self.compressing_layers(z)
        return y_logit

    def embed(self, x):
        return self.embedding_layers(x)


class DenseEdgeDecoder(torch.nn.Module):
    def __init__(self, input_dim, mlp=None):
        super().__init__()
        if mlp is None:
            self.mlp = EdgeMLP(2 * input_dim)
        else:
            self.mlp = mlp

    def forward(self, z, edge_index, sigmoid=True):
        edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        value = self.mlp(edge_features).view((edge_index.size()[1],))
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        size = z.size()
        z1 = z.repeat(size[0], 1)
        z2 = z.repeat(1, size[0]).view((size[0]*size[0], size[1]))
        edge_features = torch.cat([z1, z2], dim=-1)
        adj = self.mlp(edge_features).view((size[0], size[0]))
        return torch.sigmoid(adj) if sigmoid else adj

    def embed(self, z, edge_index):
        edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        embeddings = self.mlp.embed(edge_features)
        return embeddings


class Permute(torch.nn.Module):
    def __init__(self, permute_dims):
        super().__init__()
        self.permute_dims = permute_dims

    def forward(self, x):
        return torch.permute(x, self.permute_dims)


class NodeNN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding_layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            Permute((1, 0)),
            Conv1d(input_dim, 512, 3, padding='same'),
            Conv1d(512, 256, 3, padding='same'),
            Permute((1, 0)),
            # BatchNorm(256, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(256, 128),
            # BatchNorm(128, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(128, 64),
            # BatchNorm(64, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(64, 16),
        )
        self.compressing_layers = torch.nn.Sequential(
            # BatchNorm(16, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x, sigmoid=True):
        z = self.embedding_layers(x)
        y_logit = self.compressing_layers(z)
        return torch.sigmoid(y_logit) if sigmoid else y_logit

    def embed(self, x):
        return self.embedding_layers(x)


class DenseNodeDecoder(torch.nn.Module):
    def __init__(self, input_dim, mlp=None):
        super().__init__()
        if mlp is None:
            self.mlp = NodeNN(input_dim)
        else:
            self.mlp = mlp

    def forward(self, z, sigmoid=False):
        value = self.mlp(z).view((z.size(0),))
        return torch.sigmoid(value) if sigmoid else value

    def embed(self, z):
        return self.mlp.embed(z)