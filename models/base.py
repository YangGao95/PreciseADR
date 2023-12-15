"""
基础的PreciseADR 模型，支持异质图GNN，没有对比学习
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import Linear, Dropout, LayerNorm, functional as F
from torch_geometric.data import Batch

from torch import Tensor
from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv, GATConv
from torchmetrics.classification import MultilabelAUROC, MultilabelPrecision, MultilabelRecall

from models.utils import RetrievalHitRate, RetrievalPrecision, RetrievalRecall, FocalLoss


class BasicEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, activation=F.relu):
        super(BasicEncoder, self).__init__()
        # Implementation of Feedforward model
        self.linear1 = Linear(in_dim, hid_dim)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hid_dim, in_dim)

        self.norm1 = LayerNorm(in_dim, eps=layer_norm_eps)
        self.norm2 = LayerNorm(in_dim, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x)
        x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class HeteroEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, activation=F.relu, num_types=1):
        super(HeteroEncoder, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.num_types = num_types
        self.lins = torch.nn.ModuleList([
            BasicEncoder(in_dim, hid_dim, dropout, layer_norm_eps, activation)
            for _ in range(num_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, type_vec: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): The input features.
            type_vec (LongTensor): A vector that maps each entry to a type.
        """
        out = x.new_empty(x.size(0), self.in_dim)
        for i, lin in enumerate(self.lins):
            mask = type_vec == i
            out[mask] = lin(x[mask])
        return out


class BaseModel(nn.Module):
    def __init__(self, metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args=None):
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.in_dim_dict = in_dim_dict

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_info = n_info

        self.n_node = n_node
        self.in_dim = in_dim
        self.args = args
        self.n_layer = self.args.n_gnn
        super(BaseModel, self).__init__()

        self.build_()

    def build_(self):
        self.in_lin = nn.Linear(self.in_dim, self.hid_dim)
        encoder = nn.ModuleList()
        for _ in range(self.n_layer):
            encoder.append(BasicEncoder(self.hid_dim))
        self.encoder = nn.Sequential(*encoder)
        self.readout = nn.Linear(self.hid_dim, self.out_dim)

    def _build_x_feat(self, x, device=None):
        x = self.in_lin(x)
        x = self.encoder(x)
        return x

    def forward(self, x_dict, edge_index_dict, return_hidden=False):
        bow_feat = x_dict["patient"]
        hidden = self._build_x_feat(bow_feat, bow_feat.device)
        x = self.readout(hidden)
        if return_hidden:
            return x, hidden
        else:
            return x


class HeteroMLP(nn.Module):
    def __init__(self, metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args=None):
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.in_dim_dict = in_dim_dict

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_info = n_info

        self.n_node = n_node
        self.in_dim = in_dim
        self.args = args
        self.n_layer = self.args.n_gnn

        super(HeteroMLP, self).__init__()

        self.n_mlp = args.n_mlp if hasattr(args, "n_mlp") else 1

        nn_layers = []
        for i in range(self.n_mlp):
            in_dim = in_dim if i == 0 else hid_dim
            nn_layers.append(nn.Linear(in_dim, hid_dim))
            nn_layers.append(nn.Tanh())
            nn_layers.append(nn.Dropout(args.dropout))
        self.in_lin = nn.Sequential(*nn_layers)
        self.readout = nn.Linear(self.hid_dim, self.out_dim)

    def _build_x_feat(self, x):
        x = self.in_lin(x)
        return x

    def forward(self, x_dict, edge_index_dict, ):
        bow_feat = x_dict["patient"]
        x = self._build_x_feat(bow_feat)
        x = self.readout(x)
        return x


class RGCN(HeteroMLP):
    def __init__(self, metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args=None):
        self.n_gnn = args.n_gnn
        self.n_mlp = args.n_mlp
        super(RGCN, self).__init__(metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args)

        # HGNN
        self.convs = nn.ModuleList()
        for _ in range(self.n_gnn):
            conv = HeteroConv({
                edge_type: SAGEConv(hid_dim, hid_dim)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, return_hidden=False):
        if "info_nodes" in x_dict:
            x_dict.pop("info_nodes")

        if "attrs" in x_dict:
            x_dict.pop("attrs")

        batch_size = -1
        if "batch_size" in x_dict:
            batch_size = x_dict.pop("batch_size")

        for node_type in self.node_types:
            x_dict[node_type] = self._build_x_feat(x_dict[node_type])

        res = [x_dict["patient"]]
        for i, conv in enumerate(self.convs):
            tmp_x_dict = conv(x_dict, edge_index_dict)
            for key in x_dict:
                if key not in tmp_x_dict:
                    tmp_x_dict[key] = nn.Tanh()(x_dict[key])

            res.append(x_dict["patient"])

        hidden = x_dict["patient"]

        x = self.readout(hidden)
        if return_hidden:
            return x, hidden
        else:
            return x


class HGT(RGCN):
    def __init__(self, metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args=None):
        self.n_gnn = args.n_gnn
        self.n_mlp = args.n_mlp
        super(HGT, self).__init__(metadata, in_dim_dict, hid_dim, out_dim, n_info, n_node, in_dim, args)

        # HGNN
        self.convs = nn.ModuleList()
        for _ in range(self.n_gnn):
            conv = HGTConv(hid_dim, hid_dim, metadata, 1, group='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, return_hidden=False):
        if "info_nodes" in x_dict:
            x_dict.pop("info_nodes")

        if "attrs" in x_dict:
            x_dict.pop("attrs")

        batch_size = -1
        if "batch_size" in x_dict:
            batch_size = x_dict.pop("batch_size")

        for node_type in self.node_types:
            x_dict[node_type] = self._build_x_feat(x_dict[node_type])

        res = [x_dict["patient"]]
        att_weight = None
        for i, conv in enumerate(self.convs):
            # if i == len(self.convs) - 1 and return_att:
            #     att_weight = HGT.get_att_weights(conv, x_dict, edge_index_dict)
            tmp_x_dict = conv(x_dict, edge_index_dict)
            for key in x_dict:
                if key not in tmp_x_dict:
                    tmp_x_dict[key] = x_dict[key]

            res.append(x_dict["patient"])

        hidden = x_dict["patient"]

        x = self.readout(hidden)

        if return_hidden:
            return x, hidden
        else:
            return x


model_dict = {
    "BaseModel": lambda args: BaseModel(args.metadata, args.in_dim_dict, args.hid_dim, args.out_dim,
                                        n_info=args.num_info, n_node=args.num_node,
                                        in_dim=args.in_dim, args=args),
    "HeteroMLP": lambda args: HeteroMLP(args.metadata, args.in_dim_dict, args.hid_dim, args.out_dim,
                                        n_info=args.num_info, n_node=args.num_node,
                                        in_dim=args.in_dim, args=args),

    "RGCN": lambda args: RGCN(args.metadata, args.in_dim_dict, args.hid_dim, args.out_dim,
                              n_info=args.num_info, n_node=args.num_node,
                              in_dim=args.in_dim, args=args),

    "HGT": lambda args: HGT(args.metadata, args.in_dim_dict, args.hid_dim, args.out_dim,
                            n_info=args.num_info, n_node=args.num_node,
                            in_dim=args.in_dim, args=args),
}


def model_factory(model_name, args):
    assert model_name in model_dict, f"No model:{model_name}"
    if model_name not in model_dict:
        print(f"{model_name} not in model_dict")
        return None
    return model_dict[model_name](args)


class BasicModelWrapper(LightningModule):
    def __init__(
            self,
            model_name,
            args=None,
            model_factory_func=None
    ):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        self.model_name = model_name

        if model_factory_func is None:
            self.model = model_factory(model_name, args)
        else:
            self.model = model_factory_func(model_name, args)

        self.hit_1 = RetrievalHitRate(k=1, compute_on_cpu=True)
        self.hit_2 = RetrievalHitRate(k=2, compute_on_cpu=True)
        self.hit_5 = RetrievalHitRate(k=5, compute_on_cpu=True)
        self.hit_10 = RetrievalHitRate(k=10, compute_on_cpu=True)
        self.hit_20 = RetrievalHitRate(k=20, compute_on_cpu=True)
        self.hit_50 = RetrievalHitRate(k=50, compute_on_cpu=True)

        self.val_p_k = RetrievalPrecision(k=10, compute_on_cpu=True)
        self.val_r_k = RetrievalRecall(k=10, compute_on_cpu=True)

        self.auc = MultilabelAUROC(num_labels=args.out_dim)
        self.p = MultilabelPrecision(num_labels=args.out_dim, compute_on_cpu=True)
        self.recall = MultilabelRecall(num_labels=args.out_dim, compute_on_cpu=True)

        if self.args.loss == "kl":
            self.loss_fn = torch.nn.KLDivLoss(weight=self.args.loss_weight)

        elif self.args.loss == "focal":
            self.loss_fn = FocalLoss(gamma=self.args.gamma)

        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.args.loss_weight)

        self.train_y = []
        self.train_y_hat = []
        self.train_ids = []

        self.val_y = []
        self.val_y_hat = []
        self.val_ids = []

        self.test_y = []
        self.test_y_hat = []
        self.test_ids = []

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Tensor],
            return_att: bool = False
    ) -> Dict[NodeType, Tensor]:
        return self.model(x_dict, edge_index_dict)

    def common_step(self, batch: Batch, return_att=False) -> Tuple[Tensor, Tensor]:
        batch_size = batch['patient'].batch_size
        x_dict = {"batch_size": batch_size}
        for n_type in batch.node_types:
            x_dict[n_type] = batch[n_type].bow_feat

        y = batch['patient'].y[:batch_size]
        y_hat = self(x_dict, batch.edge_index_dict)[:batch_size]
        return y_hat, y

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        y_hat, y = self.common_step(batch)

        if self.args.loss == "kl":
            y = y / y.sum(dim=-1, keepdims=True)
            y_hat = F.log_softmax(y_hat, dim=-1)

        loss = self.loss_fn(y_hat, y)
        self.auc(y_hat, y.long())
        self.log(f'train_auc', self.auc, prog_bar=True, on_epoch=True)
        del y_hat, y
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        torch.cuda.empty_cache()
        super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        # batch.cpu()
        del batch, outputs
        torch.cuda.empty_cache()

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int):
        y_hat, y = self.common_step(batch)
        # 0,0,0,1,1,1, index of label, 每个标签属于哪个节点
        # indexes = torch.arange(y.size(0)).unsqueeze(0).t().repeat([1, y.size(1)]).view(-1)
        indexes = batch["patient"].x[:batch['patient'].batch_size].unsqueeze(0).t().repeat([1, y.size(1)]).view(-1)

        self.val_y_hat.append(y_hat.detach().cpu().clone())
        self.val_y.append(y.detach().cpu().clone())
        self.val_ids.append(indexes.detach().cpu().clone())

        del y, y_hat, indexes

    def evaluate_metrics(self, y_list, y_hat_list, index_list, mod="val", full=False):
        # y, y_hat, indexes = torch.cat(y_list, dim=0), torch.cat(y_hat_list, dim=0), torch.cat(index_list, dim=0)
        y, y_hat, indexes = torch.cat(y_list, dim=0), torch.cat(y_hat_list, dim=0), None

        # do something with all preds
        self.auc.reset()
        self.auc(y_hat, y.long())
        self.log(f'{mod}_auc', self.auc.compute(), prog_bar=True, on_epoch=True)

        self.hit_1(y_hat, y, indexes=indexes)
        self.log(f'{mod}_hit_1', self.hit_1.value, prog_bar=True, on_epoch=True)

        if full:
            self.hit_2(y_hat, y, indexes=indexes)
            self.hit_5(y_hat, y, indexes=indexes)

            self.log(f'{mod}_hit_2', self.hit_2.value, prog_bar=True, on_epoch=True)
            self.log(f'{mod}_hit_5', self.hit_5.value, prog_bar=True, on_epoch=True)

            self.hit_10(y_hat, y, indexes=indexes)
            self.log(f'{mod}_hit_10', self.hit_10.value, prog_bar=True, on_epoch=True)

        # free memory
        y_list.clear()
        y_hat_list.clear()
        index_list.clear()

    def on_train_epoch_end(self):
        # self.evaluate_metrics(self.train_y, self.train_y_hat, self.train_ids, mod="train")
        self.train_y.clear(), self.train_y_hat.clear(), self.train_ids.clear()
        super().on_train_epoch_end()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        self.evaluate_metrics(self.val_y, self.val_y_hat, self.val_ids)
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        self.evaluate_metrics(self.test_y, self.test_y_hat, self.test_ids, mod="test", full=True)
        super().on_test_epoch_end()

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int):
        y_hat, y = self.common_step(batch)

        indexes = batch["patient"].x[:batch['patient'].batch_size].unsqueeze(0).t().repeat([1, y.size(1)]).view(-1)

        self.test_y_hat.append(y_hat.detach().cpu().clone())
        self.test_y.append(y.detach().cpu().clone())
        self.test_ids.append(indexes.detach().cpu().clone())

        del y, y_hat, indexes

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.use_scheduler:
            opt_sche = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.args.max_epochs)
            opt_sche = torch.optim.lr_scheduler.OneCycleLR(opt, self.args.lr * 10, total_steps=self.args.max_epochs)
            return {"optimizer": opt, "lr_scheduler": opt_sche}
        else:
            return opt

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        # y_hat, y, att_dict, edge_index_dict = self.common_step(batch, return_att=False)
        y_hat, y = self.common_step(batch)
        return y_hat, y

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for n_type in batch.node_types:
            batch[n_type].bow_feat = batch[n_type].bow_feat.to(device)
        batch["patient"].y = batch["patient"].y.to(device)
        edge_index_dict = {k: v.to(device) for k, v in batch.edge_index_dict.items()}
        batch.edge_index_dict = edge_index_dict
        return batch
