import glob
import os.path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from gnns import AttentionLeEncoder, DenseNodeDecoder, AutomaticWeightedLoss, \
    CrossStitchedGAE, CrossStitchedVGAE, SingleTaskGAE, SingleTaskVGAE, DenseSingleTaskDecoder, NodeHead, EdgeHead, GraphSageEncoder, VariationalGraphSageEncoder
from tqdm.auto import tqdm
from train_utils import EarlyStopper, save_model, load_model
from nn_data import RemoveSelfLooping, easy_to_device
from nn_data import ShortDistanceNegSampler, colwise_in
from post_process import remove_short_distance_loops
from configs import DEVICE, LOADER_WORKER
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError
from torch.utils.data import RandomSampler
from configs import CompilationConfigs, TrainConfigs, SEED
from schickit.utils import create_bin_df, get_chrom_sizes
from layers import torch_algrelmax
from nn_data import short_dist_neg_sampling, filter_range
EPS = 1e-7
from gnns import DenseEdgeDecoder, MultiTaskVGAE
from torch_geometric.utils import sort_edge_index

# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)


def up_lower_tria_vote(df):
    df = df[df['chrom1'] == df['chrom2']]
    triu_df = df[df['y1'] > df['x1']]
    tril_df = df[df['y1'] < df['x1']]
    tril_df = tril_df.rename(columns={'x1': 'y1', 'x2': 'y2', 'y1': 'x1', 'y2': 'x2'})
    tril_df = tril_df[['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2', 'proba']]
    df = pd.concat([triu_df, tril_df])
    df = df.groupby(['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'], as_index=False, sort=False).mean()
    return df


def deduplicate_bedpe_dir(bedpe_dir):
    bedpe_files_paths = glob.glob(os.path.join(bedpe_dir, '*.csv'))
    for bedpe_path in bedpe_files_paths:
        df = pd.read_csv(bedpe_path, sep='\t', header=0, index_col=False, dtype={'proba': 'float'})
        try:
            df = df.groupby(['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'], sort=True, as_index=False).mean()
        except:
            print(df)
        df.to_csv(bedpe_path, sep='\t', header=True, index=False)



class MultitaskFeatureCaller(object):
    def __init__(self, run_id, chroms, model_path, num_feature, alpha, beta, log_dir=None):
        self.run_id = run_id
        self.chroms = chroms
        self.model_path = model_path
        self.log_dir = log_dir

        self.in_channels, self.latent_channels = num_feature, CompilationConfigs.latent_channels
        self.learning_rate = CompilationConfigs.learning_rate
        # self.kl_coef = CompilationConfigs.kl_coef
        self.weight_decay = CompilationConfigs.weight_decay
        self.alpha = alpha
        self.beta = beta
        self.vgae = MultiTaskVGAE(
            VariationalGraphSageEncoder(self.in_channels, self.latent_channels),
            DenseEdgeDecoder(self.latent_channels),
            DenseEdgeDecoder(self.latent_channels)
        ).to(DEVICE)

    def load_model(self):
        checkpoint = torch.load(self.model_path)
        self.vgae.load_state_dict(checkpoint['model_state_dict'])

    def loop_classification_loss(self, decoder, z, data, neg_loop_index=None):

        pos_loss = -torch.log(
            decoder(z, data.edge_label_index, sigmoid=True) + EPS).mean()
        if neg_loop_index is None:
            neg_loop_index = short_dist_neg_sampling(data.edge_label_index, data.edge_index, data.num_nodes)
        neg_loss = -torch.log(1 -
                              decoder(z, neg_loop_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def recon_loss(self, decoder, z, data, neg_edge_index=None):

        edge_index = filter_range(data.edge_index)
        if edge_index.size(1) > 20000:
            # Randomly select 20000 edges
            idx = torch.randperm(edge_index.size(1))[:20000]
            edge_index = edge_index[:, idx]
            edge_index = sort_edge_index(edge_index)

        pos_loss = -torch.log(
            decoder(z, edge_index, sigmoid=True) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = short_dist_neg_sampling(edge_index, edge_index, data.num_nodes)
        neg_loss = -torch.log(1 -
                              decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    @torch.no_grad()
    def get_evaltime_edgewise_loss(self, pos_pred, neg_pred):

        pos_loss = -torch.log(pos_pred + EPS).mean()
        neg_loss = -torch.log(1 - neg_pred + EPS).mean()
        return pos_loss + neg_loss

    def train_vgae_batch(self, data, vgae, optimizer, device):
        vgae.train()
        data = data.to(device)
        optimizer.zero_grad()
        z = vgae.encode(data.x, data.edge_index)

        loss_loop = self.loop_classification_loss(vgae.decoder, z, data)
        loss_recon = self.recon_loss(vgae.recon_decoder, z, data)

        kl_coef = 1 / data.num_nodes
        kl_loss = kl_coef * vgae.kl_loss()

        loss = self.alpha * loss_loop + self.beta * loss_recon + kl_loss

        loss.backward()
        optimizer.step()
        return loss.detach().item(), loss_loop.detach().item(), loss_recon.detach().item()

    @torch.no_grad()
    def approx_vgae_evaluate(self, loader, vgae, device, recon_label):
        ap_metric = BinaryAveragePrecision()
        auroc_metric = BinaryAUROC()
        assert recon_label in ['contact', 'loop']
        attrs_to_remove = ['chrom_name', 'cell_name', 'cell_type', 'edge_weights']
        if recon_label == 'contact':
            attrs_to_remove.append('edge_label_index')
        vgae.eval()
        losses = []
        loop_losses = []
        recon_losses = []
        roc_auc_list = []
        ap_list = []
        for batch in loader:
            batch = easy_to_device(batch, device, attrs_to_remove)
            z = vgae.encode(batch.x, batch.edge_index)
            loop_labels = batch.edge_label_index
            loop_pos_pred = vgae.decode(z, loop_labels, sigmoid=True)

            batch.neg_loop_index = short_dist_neg_sampling(
                loop_labels, batch.edge_index, batch.num_nodes, fix_seed=SEED
            )

            loop_neg_pred = vgae.decode(z, batch.neg_loop_index, sigmoid=True)
            loop_pos_y = z.new_ones(loop_labels.size(1))
            loop_neg_y = z.new_zeros(batch.neg_loop_index.size(1))
            loop_y_label = torch.cat([loop_pos_y, loop_neg_y])
            loop_y_pred = torch.cat([loop_pos_pred, loop_neg_pred])
            ap = ap_metric(loop_y_pred, loop_y_label.int())
            auroc = auroc_metric(loop_y_pred, loop_y_label.int())
            loop_cls_loss = self.get_evaltime_edgewise_loss(loop_pos_pred, loop_neg_pred)


            recon_index = filter_range(batch.edge_index)
            if recon_index.size(1) > 20000:
                idx = torch.randperm(recon_index.size(1))[:20000]
                recon_index = recon_index[:, idx]
                recon_index = sort_edge_index(recon_index)
            recon_pos_pred = vgae.recon_decoder(z, recon_index, sigmoid=True)
            batch.neg_edge_index = short_dist_neg_sampling(
                recon_index, recon_index, batch.num_nodes, fix_seed=SEED
            )
            recon_neg_pred = vgae.recon_decoder(z, batch.neg_edge_index, sigmoid=True)
            recon_loss = self.get_evaltime_edgewise_loss(recon_pos_pred, recon_neg_pred)


            loss = self.alpha * loop_cls_loss + self.beta * recon_loss

            losses.append(loss.detach().item())
            loop_losses.append(loop_cls_loss.detach().item())
            recon_losses.append(recon_loss.detach().item())

            # kl_losses.append(kl_loss.detach().item())
            roc_auc_list.append(auroc.detach().item())
            ap_list.append(ap.detach().item())
        mean_auroc = np.nanmean(np.asarray(roc_auc_list))
        mean_ap = np.nanmean(np.asarray(ap_list))
        # print(np.mean(np.asarray(kl_losses)))
        return mean_auroc, mean_ap, np.mean(np.asarray(losses)), np.mean(np.asarray(loop_losses)), np.mean(np.asarray(recon_losses))

    def train(self, train_set, val_set, bs=1, epochs=100):
        print('Training the loop classifier...')
        loop_optimizer = torch.optim.Adam(
            self.vgae.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay)
        train_loader, val_loader = \
            DataLoader(
                train_set, bs, num_workers=LOADER_WORKER,
                pin_memory=False, exclude_keys=['edge_weights', 'cell_name', 'chrom_name', 'cell_type'],
                sampler=RandomSampler(
                    train_set, replacement=True,
                    num_samples=TrainConfigs.train_samples_per_epoch
                ),
            ), \
            DataLoader(
                val_set, bs, num_workers=LOADER_WORKER, pin_memory=False,
                # sampler=RandomSampler(
                #     val_set, replacement=True,
                #     num_samples=100
                # )
            )
        early_stopper = EarlyStopper(patience=5)
        for epoch in range(1, epochs + 1):
            epoch_loss_list = []
            epoch_loop_loss_list = []
            epoch_recon_loss_list = []
            for i, batch in enumerate(tqdm(train_loader, leave=False, position=0, desc='Epoch {}'.format(epoch))):
                loss, loop_loss, recon_loss = self.train_vgae_batch(batch, self.vgae, loop_optimizer, DEVICE)
                epoch_loss_list.append(loss)
                epoch_loop_loss_list.append(loop_loss)
                epoch_recon_loss_list.append(recon_loss)
            auc, ap, val_loss, val_loop_loss, val_recon_loss = self.approx_vgae_evaluate(
                val_loader, self.vgae, DEVICE, 'loop'
            )
            mean_loss = np.array(epoch_loss_list).mean()
            print(
                f'\t Epoch: {epoch:03d}, train loss: {mean_loss:.4f}, '
                f'train Loop Loss: {np.array(epoch_loop_loss_list).mean():.4f}, '
                f'train Recon Loss: {np.array(epoch_recon_loss_list).mean():.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Loop Loss: {val_loop_loss:.4f}, '
                f'Val Recon Loss: {val_recon_loss:.4f}, '
                f'Val AUC: {auc:.4f}, Val AP: {ap:.4f} '
            )
            if val_loss < early_stopper.min_validation_loss:
                save_model(epoch, self.vgae, loop_optimizer, loss, self.model_path)
                print('Checkpoint saved. ')
            if early_stopper.early_stop(val_loss):
                break

    @torch.no_grad()
    def predict(self, loop_dir, test_set, device, loop_threshold, resolution=10000, progress_bar=True, output_embedding=None):
        bs = 1
        vgae = self.vgae.to(device)
        vgae.eval()
        os.makedirs(loop_dir, exist_ok=False)
        loader = DataLoader(test_set, bs, num_workers=LOADER_WORKER, pin_memory=False, )
        attrs_to_remove = ['chrom_name', 'cell_name', 'cell_type', 'edge_weights', 'edge_label_index']

        if output_embedding is not None:
            os.makedirs(output_embedding, exist_ok=False)

        print('Predicting...')
        for batch in (tqdm(loader) if progress_bar else loader):
            batch = easy_to_device(batch, device, attrs_to_remove)
            z = vgae.encode(batch.x, batch.edge_index)
            preds = vgae.decode(z, batch.edge_index, sigmoid=True)
            preds = preds.detach().cpu().numpy()
            edges = batch.edge_index.cpu().numpy()
            df = self.convert_batch_loop_preds_to_df(preds, edges, batch.chrom_name[0], resolution)
            assert len(df) == edges.shape[1]
            df = up_lower_tria_vote(df)
            assert len(df) == edges.shape[1] // 2
            df = remove_short_distance_loops(df)
            df = df[df['proba'] >= loop_threshold]
            df = df.drop_duplicates(subset=['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'])  # Do we really need to do this?
            df = df.reset_index(drop=True)
            short_cell_name = batch.cell_name[0].split('/')[-1]
            cell_csv_path = os.path.join(loop_dir, f'{short_cell_name}.csv')
            df.to_csv(
                cell_csv_path, sep='\t', header=not os.path.exists(cell_csv_path),
                index=False, mode='a', float_format='%.5f'
            )
            if output_embedding is not None:
                current_chr = batch.chrom_name[0]
                emb_path = os.path.join(output_embedding, f'{short_cell_name}_{current_chr}.npy')
                np.save(emb_path, z.detach().cpu().numpy())
        print('Done!')


    def convert_batch_loop_preds_to_df(self, preds, edges, chrom_name, resolution):
        proba_vector = preds
        x1_vector = edges[0, :] * resolution
        x2_vector = x1_vector + resolution
        y1_vector = edges[1, :] * resolution
        y2_vector = y1_vector + resolution
        chroms = [chrom_name] * len(proba_vector)
        df = pd.DataFrame({
            'chrom1': chroms, 'x1': x1_vector, 'x2': x2_vector, 'chrom2': chroms,
            'y1': y1_vector, 'y2': y2_vector, 'proba': proba_vector
        })
        df = df.astype({'x1': 'int', 'x2': 'int', 'y1': 'int', 'y2': 'int'})
        return df




class CrossStitchFeatureCaller(object):
    def __init__(self, run_id, chroms, model_path, num_feature, auto_weighted_loss, alpha, beta, a, b, freeze, log_dir=None):
        self.run_id = run_id
        self.chroms = chroms
        self.model_path = model_path
        self.log_dir = log_dir

        self.in_channels, self.latent_channels = num_feature, CompilationConfigs.latent_channels
        self.head_input_channels = CompilationConfigs.head_input_channels
        self.learning_rate = CompilationConfigs.learning_rate
        # self.kl_coef = CompilationConfigs.kl_coef
        self.weight_decay = CompilationConfigs.weight_decay
        self.auto_weighted_loss = auto_weighted_loss
        if self.auto_weighted_loss:
            self.auto_loss_addition = AutomaticWeightedLoss(2)

        self.alpha = alpha  # Loss = alpha * recon_loss_edge + beta * node_loss; Ignored if auto_weighted_loss is True.
        self.beta = beta

        self.cross_stitch_param_a = a  # Cross stitch parameter that controls the contribution of the task itself and other tasks
        self.cross_stitch_param_b = b
        self.freeze_cross_stitch = freeze

        self.model, self.optimizer = self.get_model_settings()
        # self.writer = SummaryWriter(log_dir='logs/tb_logs/test_logs')

    def load_model(self):
        if self.auto_weighted_loss:
            self.model, self.optimizer, *_ = load_model(
                self.model, self.optimizer, self.model_path, self.auto_loss_addition
            )
        else:
            self.model, self.optimizer, *_ = load_model(self.model, self.optimizer, self.model_path)

    def get_model_settings(self):
        loop_single_task_gae = SingleTaskVGAE(
            VariationalGraphSageEncoder(self.in_channels, self.latent_channels),
            DenseSingleTaskDecoder(self.latent_channels, self.head_input_channels)
        )
        node_single_task_gae = SingleTaskGAE(
            AttentionLeEncoder(self.in_channels, self.latent_channels),
            DenseSingleTaskDecoder(self.latent_channels, self.head_input_channels)
        )
        net = CrossStitchedVGAE(
            ['loop_pred', 'node_reg'],
            torch.nn.ModuleDict({
                'loop_pred': loop_single_task_gae,
                'node_reg': node_single_task_gae
            }),
            heads=torch.nn.ModuleDict({
                'loop_pred': EdgeHead(self.head_input_channels * 2),
                'node_reg': NodeHead(self.head_input_channels)
            }),
            encoder_stages=None,
            # decoder_stages=['decoder_layer1', 'decoder_layer2', 'decoder_layer3'],
            decoder_stages=['decoder_layer1'],
            encoder_channels={},
            # decoder_channels={'decoder_layer1': 256, 'decoder_layer2': 128, 'decoder_layer3': self.head_input_channels},
            decoder_channels={'decoder_layer1': self.head_input_channels},
            alpha=self.cross_stitch_param_a, beta=self.cross_stitch_param_b, freeze_cross_stitch=self.freeze_cross_stitch
        )

        # model = VGAE(VariationalGraphSageEncoder(in_channels, out_channels))
        net = net.to(DEVICE)
        if self.auto_weighted_loss:
            opt = torch.optim.Adam([
                {'params': net.parameters(), 'weight_decay': self.weight_decay},
                {'params': self.auto_loss_addition.parameters(), 'weight_decay': 0}
            ], lr = self.learning_rate)
        else:
            opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        return net, opt

    def train_batch(self, data, model, optimizer, device):
        model.train()
        data = data.to(device)
        optimizer.zero_grad()
        latent = model.encode(data.x, data.edge_index)
        z = model.compute_backbone_output(latent)
        losses = model.get_losses(z, data, ls=CompilationConfigs.label_smoothing)
        edge_kl_loss = model.get_kl_loss_of_task('loop_pred')
        loss_edge = losses['loop_pred']
        loss_edge += (1 / data.num_edges) * edge_kl_loss
        loss_node = losses['node_reg']

        # if kl_coef is None:
        #     kl_coef = 1 / data.num_nodes
        # kl_loss = 0
        # kl_losses = model.get_kl_losses()
        # for l in kl_losses:
        #     kl_loss += kl_coef * kl_losses[l]
        if self.auto_weighted_loss:
            loss = self.auto_loss_addition(loss_edge, loss_node) # + kl_loss
        else:
            loss = self.alpha * loss_edge + self.beta * loss_node # + kl_loss
        loss.backward()
        optimizer.step()
        return loss.detach().item(), loss_edge.detach().item(), loss_node.detach().item()

    @torch.no_grad()
    def approx_evaluate_all(self, loader, model, device, edge_recon_label, kl_coef=None):
        ap_metric = BinaryAveragePrecision()
        auroc_metric = BinaryAUROC()
        mse_metric = MeanSquaredError().to(device)
        mape_metric = MeanAbsolutePercentageError().to(device)
        assert edge_recon_label in ['contact', 'loop']
        attrs_to_remove = ['chrom_name', 'cell_name', 'cell_type', 'edge_weights']
        if edge_recon_label == 'contact':
            attrs_to_remove.append('edge_label_index')
        model.eval()
        loss_edge_list = []
        loss_node_list = []
        losses = []
        # kl_losses = []
        roc_auc_list = []
        ap_list = []
        mse_list = []
        mape_list = []
        for batch in loader:
            batch = easy_to_device(batch, device, attrs_to_remove)
            z = model.encode(batch.x, batch.edge_index)
            backbone_output = model.compute_backbone_output(z)

            # Evaluate performance on the edge task.
            labels = batch.edge_index if edge_recon_label == 'contact' else batch.edge_label_index
            pos_pred = model.compute_head_from_backbone_output(
                {'loop_pred': backbone_output['loop_pred']}, {'loop_pred': labels}
            )['loop_pred']
            neg_sampler = ShortDistanceNegSampler(fix_seed=SEED)
            batch = neg_sampler(batch)
            neg_pred = model.compute_head_from_backbone_output(
                {'loop_pred': backbone_output['loop_pred']}, {'loop_pred': batch.neg_edge_index}
            )['loop_pred']
            pos_y = z['loop_pred'].new_ones(labels.size(1))
            neg_y = z['loop_pred'].new_zeros(batch.neg_edge_index.size(1))
            y_label = torch.cat([pos_y, neg_y])
            y_pred = torch.cat([pos_pred, neg_pred])
            ap = ap_metric(y_pred, y_label.int())
            auroc = auroc_metric(y_pred, y_label.int())
            loss_edge = model.get_evaltime_loss_loop_pred(pos_pred, neg_pred, ls=CompilationConfigs.label_smoothing)

            # Evaluate performance on the node task.
            tad_pred = model.compute_head_from_backbone_output(
                {'node_reg': backbone_output['node_reg']}, {}
            )['node_reg']
            loss_node = model.get_evaltime_loss_node_reg(tad_pred, batch.tad_label)
            mse = mse_metric(tad_pred, batch.tad_label)
            mape = mape_metric(tad_pred, batch.tad_label)

            # if kl_coef is None:
            #     kl_coef = 1 / batch.num_nodes
            # kl_loss = 0
            # task_kl_losses = model.get_kl_losses()
            # for l in task_kl_losses:
            #     kl_loss += kl_coef * task_kl_losses[l]
            # if self.auto_weighted_loss:
            #     loss = self.auto_loss_addition(loss_edge, loss_node) # + kl_loss
            # else:
            loss = self.alpha * loss_edge + self.beta * loss_node # + kl_loss
            losses.append(loss.detach().item())
            loss_edge_list.append(loss_edge.detach().item())
            loss_node_list.append(loss_node.detach().item())
            # print(task_kl_losses['node_reg'].detach().item())
            # print(task_kl_losses['loop_pred'].detach().item())
            # kl_losses.append(kl_loss.detach().item())
            roc_auc_list.append(auroc.detach().item())
            ap_list.append(ap.detach().item())
            mse_list.append(mse.detach().item())
            mape_list.append(mape.detach().item())
        mean_auroc = np.nanmean(np.asarray(roc_auc_list))
        mean_ap = np.nanmean(np.asarray(ap_list))
        mean_mse = np.nanmean(np.asarray(mse_list))
        mean_mape = np.nanmean(np.asarray(mape_list))
        return mean_auroc, mean_ap, mean_mse, mean_mape, \
            np.mean(np.asarray(losses)), np.mean(np.asarray(loss_edge_list)), np.mean(np.asarray(loss_node_list))


    def train(self, train_set, val_set, bs=1, epochs=100):
        train_loader, val_loader = \
            DataLoader(
                train_set, bs, num_workers=LOADER_WORKER,
                pin_memory=False, exclude_keys=['edge_weights', 'cell_name', 'chrom_name', 'cell_type'],
                sampler=RandomSampler(
                    train_set, replacement=True,
                    num_samples=TrainConfigs.train_samples_per_epoch
                ),
            ), \
            DataLoader(
                val_set, bs, num_workers=LOADER_WORKER, pin_memory=False,
                # sampler=RandomSampler(
                #     val_set, replacement=True,
                #     num_samples=100
                # )
            )
        early_stopper = EarlyStopper(patience=5)
        for epoch in range(1, epochs + 1):
            epoch_loss_list = []
            epoch_edge_loss_list = []
            epoch_node_loss_list = []
            for i, batch in enumerate(tqdm(train_loader, leave=False, position=0, desc='Epoch {}'.format(epoch))):
                step = (epoch - 1) * len(train_loader) + i
                loss, loss_edge, loss_node = self.train_batch(batch, self.model, self.optimizer, DEVICE)
                epoch_loss_list.append(loss)
                epoch_edge_loss_list.append(loss_edge)
                epoch_node_loss_list.append(loss_node)
            auc, ap, mse, mape, val_loss, val_loss_edge, val_loss_node = self.approx_evaluate_all(
                val_loader, self.model, DEVICE, 'loop'
            )
            mean_loss = np.array(epoch_loss_list).mean()
            mean_loss_edge = np.array(epoch_edge_loss_list).mean()
            mean_loss_node = np.array(epoch_node_loss_list).mean()
            print(self.model.cross_stitch_decoder['decoder_layer1'].cross_stitch_unit['node_reg']['node_reg'].param.detach().cpu().numpy().mean())
            print(self.model.cross_stitch_decoder['decoder_layer1'].cross_stitch_unit['node_reg']['loop_pred'].param.detach().cpu().numpy().mean())
            print(
                f'\t Epoch: {epoch:03d}, train loss: {mean_loss:.4f}, '
                f'train edge loss: {mean_loss_edge:.4f}, train node loss: {mean_loss_node:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Edge Loss: {val_loss_edge:.4f}, Val Node Loss: {val_loss_node:.4f}, '
                f'Val AUC: {auc:.4f}, Val AP: {ap:.4f}, '
                f'Val MAPE: {mape: .4f}, Val MSE: {mse:.4f}'
            )
            if val_loss < early_stopper.min_validation_loss:
                if self.auto_weighted_loss:
                    save_model(epoch, self.model, self.optimizer, loss, self.model_path, self.auto_loss_addition)
                else:
                    save_model(epoch, self.model, self.optimizer, loss, self.model_path)
                print('Checkpoint saved. ')
            if early_stopper.early_stop(val_loss):
                break
        # if self.writer is not None:
        #     self.writer.flush()
        #     self.writer.close()

    @torch.no_grad()
    def predict(self, loop_dir, tad_dir, test_set, device, loop_threshold, chrom_sizes_path, resolution=10000, progress_bar=True):
        bs = 1     # The current version only supports batch size 1.
        model = self.model.to(device)
        self.model.eval()
        os.makedirs(loop_dir, exist_ok=False)
        os.makedirs(tad_dir, exist_ok=False)
        loader = DataLoader(test_set, bs, num_workers=LOADER_WORKER, pin_memory=False, )
        chrom_sizes = get_chrom_sizes(chrom_sizes_path)
        attrs_to_remove = ['chrom_name', 'cell_name', 'cell_type', 'edge_weights', 'edge_label_index', 'tad_label']
        print('Predicting...')
        for batch in (tqdm(loader) if progress_bar else loader):
            batch = easy_to_device(batch, device, attrs_to_remove)
            z = model.encode(batch.x, batch.edge_index)
            backbone_output = model.compute_backbone_output(z)
            loop_pred = model.compute_head_from_backbone_output(
                {'loop_pred': backbone_output['loop_pred']}, {'loop_pred': batch.edge_index}
            )['loop_pred']
            tad_pred = model.compute_head_from_backbone_output(
                {'node_reg': backbone_output['node_reg']}, {}
            )['node_reg']
            loop_pred = loop_pred.detach().cpu().numpy()
            tad_pred = tad_pred.detach().cpu().numpy()
            edges = batch.edge_index.cpu().numpy()

            loop_df = self.convert_batch_loop_preds_to_df(loop_pred, edges, batch.chrom_name[0], resolution)
            assert len(loop_df) == edges.shape[1]
            loop_df = up_lower_tria_vote(loop_df)
            assert len(loop_df) == edges.shape[1] // 2
            loop_df = remove_short_distance_loops(loop_df)
            loop_df = loop_df[loop_df['proba'] >= loop_threshold]
            loop_df = loop_df.drop_duplicates(subset=['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'])   # Do we really need to do this?
            loop_df = loop_df.reset_index(drop=True)

            tad_df = self.convert_batch_tad_preds_to_df(tad_pred, batch.chrom_name[0], chrom_sizes, resolution)

            short_cell_name = batch.cell_name[0].split('/')[-1]
            loop_csv_path = os.path.join(loop_dir, f'{short_cell_name}.csv')
            loop_df.to_csv(
                loop_csv_path, sep='\t', header=not os.path.exists(loop_csv_path),
                index=False, mode='a', float_format='%.5f'
            )
            tad_csv_path = os.path.join(tad_dir, f'{short_cell_name}.csv')
            tad_df.to_csv(
                tad_csv_path, sep='\t', header=not os.path.exists(tad_csv_path),
                index=False, mode='a', float_format='%.5f'
            )
        print('Done!')

    # @torch.no_grad()
    # def predict(self, loop_dir, tad_dir, test_set, device, loop_threshold, chrom_sizes_path, window,
    #             lower_dist, upper_dist, resolution=10000, progress_bar=True):
    #     bs = 1     # The current version only supports batch size 1.
    #     model = self.model.to(device)
    #     self.model.eval()
    #     os.makedirs(loop_dir, exist_ok=False)
    #     os.makedirs(tad_dir, exist_ok=False)
    #     loader = DataLoader(test_set, bs, num_workers=LOADER_WORKER, pin_memory=False, )
    #     chrom_sizes = get_chrom_sizes(chrom_sizes_path)
    #     attrs_to_remove = ['chrom_name', 'cell_name', 'cell_type', 'edge_weights', 'edge_label_index', 'tad_label']
    #     lower, upper = lower_dist // resolution, upper_dist // resolution + 1
    #     print('Predicting...')
    #     for batch in (tqdm(loader) if progress_bar else loader):
    #         num_nodes = batch.num_nodes
    #         short_cell_name = batch.cell_name[0].split('/')[-1]
    #         loop_csv_path = os.path.join(loop_dir, f'{short_cell_name}.csv')
    #         tad_csv_path = os.path.join(tad_dir, f'{short_cell_name}.csv')
    #         batch = easy_to_device(batch, device, attrs_to_remove)
    #         z = model.encode(batch.x, batch.edge_index)
    #         backbone_output = model.compute_backbone_output(z)
    #         tad_pred = model.compute_head_from_backbone_output(
    #             {'node_reg': backbone_output['node_reg']}, {}
    #         )['node_reg']
    #         sc_tad_index = torch_algrelmax(-tad_pred, window).view(-1)
    #         sc_tad_index_tad_scores = tad_pred[sc_tad_index]
    #         # Select the lowest 150 according to the scores.
    #         if len(sc_tad_index) > 150:
    #             sc_tad_index = sc_tad_index[sc_tad_index_tad_scores.argsort()[:150]]
    #         sc_tad_index = sc_tad_index.sort()[0]
    #         tad_pred = tad_pred.detach().cpu().numpy()
    #         tad_df = self.convert_batch_tad_preds_to_df(tad_pred, batch.chrom_name[0], chrom_sizes, resolution)
    #         tad_df.to_csv(
    #             tad_csv_path, sep='\t', header=not os.path.exists(tad_csv_path),
    #             index=False, mode='a', float_format='%.5f'
    #         )
    #
    #         possible_edges = batch.edge_index
    #         extended_possible_edges = torch.combinations(sc_tad_index).t()
    #         assert torch.all(extended_possible_edges[1] > extended_possible_edges[0])
    #         # Make sure that extended possible edges are within lower and upper distance.
    #         extended_possible_edges = extended_possible_edges[:,
    #             ((extended_possible_edges[1] - extended_possible_edges[0] >= lower) &
    #             (extended_possible_edges[1] - extended_possible_edges[0] <= upper))
    #         ]
    #         extended_possible_edges = extended_possible_edges[
    #             :, torch.logical_not(colwise_in(extended_possible_edges, possible_edges, num_nodes))
    #         ]
    #         possible_edges = torch.cat([possible_edges, extended_possible_edges], dim=1)
    #         possible_edges_np = possible_edges.cpu().numpy()
    #         assert len(set(zip(possible_edges_np[0, :], possible_edges_np[1, :]))) == possible_edges_np.shape[1]
    #         possible_edges = torch.from_numpy(possible_edges_np).to(device)
    #
    #         loop_pred = model.compute_head_from_backbone_output(
    #             {'loop_pred': backbone_output['loop_pred']}, {'loop_pred': possible_edges}
    #         )['loop_pred']
    #         loop_pred = loop_pred.detach().cpu().numpy()
    #
    #         loop_df = self.convert_batch_loop_preds_to_df(loop_pred, possible_edges_np, batch.chrom_name[0], resolution)
    #         assert len(loop_df) == possible_edges_np.shape[1]
    #         loop_df = remove_short_distance_loops(loop_df)
    #         loop_df = loop_df[loop_df['proba'] >= loop_threshold]
    #         loop_df = loop_df.drop_duplicates(
    #             subset=['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'])  # Do we really need to do this?
    #         loop_df = loop_df.reset_index(drop=True)
    #         loop_df.to_csv(
    #             loop_csv_path, sep='\t', header=not os.path.exists(loop_csv_path),
    #             index=False, mode='a', float_format='%.5f'
    #         )
    #     print('Done!')

    def convert_batch_tad_preds_to_df(self, preds, chrom_name, chrom_sizes, resolution):
        df = create_bin_df(chrom_sizes, resolution, [chrom_name])
        df['score'] = preds
        df = df.reset_index(drop=True)
        return df

    def convert_batch_loop_preds_to_df(self, preds, edges, chrom_name, resolution):
        proba_vector = preds
        x1_vector = edges[0, :] * resolution
        x2_vector = x1_vector + resolution
        y1_vector = edges[1, :] * resolution
        y2_vector = y1_vector + resolution
        chroms = [chrom_name] * len(proba_vector)
        df = pd.DataFrame({
            'chrom1': chroms, 'x1': x1_vector, 'x2': x2_vector, 'chrom2': chroms,
            'y1': y1_vector, 'y2': y2_vector, 'proba': proba_vector
        })
        df = df.astype({'x1': 'int', 'x2': 'int', 'y1': 'int', 'y2': 'int'})
        return df

