from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv, GATConv

from models.base import model_dict
from .base import BasicModelWrapper
from .info_nce import info_nce


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
        # AE trans
        nn_layers = []
        for i in range(self.n_mlp):
            in_dim = in_dim if i == 0 else hid_dim
            nn_layers.append(nn.Linear(in_dim, hid_dim))
            nn_layers.append(nn.Tanh())
            nn_layers.append(nn.Dropout(args.dropout))
        self.se_trans = nn.Sequential(*nn_layers)

        # used for Contrastive Learning
        self.cl_lin = nn.Linear(self.in_dim, self.hid_dim)

        # HGNN
        self.convs = nn.ModuleList()
        for _ in range(self.n_gnn):
            conv = HeteroConv({
                edge_type: SAGEConv(hid_dim, hid_dim)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.edge_drop = nn.Dropout(args.edge_drop if hasattr(args, "edge_drop") else 0.9)
        self.aug_add = nn.Dropout(args.aug_add if hasattr(args, "aug_add") else 0.1)

    def build_aug(self, x):
        aug_x = torch.zeros_like(x)
        # 将aug_x中 1% 的元素 变为 1
        aug_mask = torch.ones_like(x)
        # 被dropout的元素设为 1， val 的时候，不产生影响
        aug_mask = (self.aug_add(aug_mask) < 1)
        aug_x[aug_mask] = 1
        return x + aug_x

    def forward(self, x_dict, edge_index_dict, return_hidden=False):
        if "info_nodes" in x_dict:
            x_dict.pop("info_nodes")

        if "attrs" in x_dict:
            x_dict.pop("attrs")

        batch_size = -1
        if "batch_size" in x_dict:
            batch_size = x_dict.pop("batch_size")

        x_aug = self.build_aug(x_dict["patient"][:batch_size])
        hid_aug = self.cl_lin(x_aug)
        hid_assis = self.cl_lin(x_dict["patient"][:batch_size])

        for node_type in self.node_types:
            if node_type == "SE":
                x_dict[node_type] = self.se_trans(x_dict[node_type])
            else:
                x_dict[node_type] = self._build_x_feat(x_dict[node_type])

        res = [x_dict["patient"]]

        if self.args.add_SE:
            edge_index = edge_index_dict[("patient", "p_se", "SE")]
            edge_mask = torch.ones(edge_index.size(1))
            edge_mask = self.edge_drop(edge_mask).bool()
            edge_index = edge_index[:, edge_mask]
            edge_index_dict[("patient", "p_se", "SE")] = edge_index
            edge_index_dict[("SE", "rev_p_se", "patient")] = edge_index[[1, 0], :]

        for i, conv in enumerate(self.convs):
            tmp_x_dict = conv(x_dict, edge_index_dict)
            for key in x_dict:
                if key not in tmp_x_dict:
                    tmp_x_dict[key] = nn.Tanh()(x_dict[key])

            res.append(x_dict["patient"])

        hidden = x_dict["patient"][:batch_size]

        x = self.readout(hidden + hid_assis)
        if return_hidden:
            return x, hidden, hid_assis, hid_aug
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

        x_aug = self.build_aug(x_dict["patient"][:batch_size])
        hid_aug = self.cl_lin(x_aug)
        hid_assis = self.cl_lin(x_dict["patient"][:batch_size])

        for node_type in self.node_types:
            if node_type == "SE":
                x_dict[node_type] = self.se_trans(x_dict[node_type])
            else:
                x_dict[node_type] = self._build_x_feat(x_dict[node_type])

        res = [x_dict["patient"]]
        for i, conv in enumerate(self.convs):
            tmp_x_dict = conv(x_dict, edge_index_dict)
            for key in x_dict:
                if key not in tmp_x_dict:
                    tmp_x_dict[key] = x_dict[key]

            res.append(x_dict["patient"])

        hidden = x_dict["patient"][:batch_size]

        x = self.readout(hidden + hid_assis)

        if return_hidden:
            return x, hidden, hid_assis, hid_aug
        else:
            return x


cur_model_dict = {"HeteroMLP": lambda args: HeteroMLP(args.metadata, args.in_dim_dict, args.hid_dim, args.out_dim,
                                                      n_info=args.num_info, n_node=args.num_node,
                                                      in_dim=args.in_dim, args=args),

                  "RGCN": lambda args: RGCN(args.metadata, args.in_dim_dict, args.hid_dim, args.out_dim,
                                            n_info=args.num_info, n_node=args.num_node,
                                            in_dim=args.in_dim, args=args),

                  "HGT": lambda args: HGT(args.metadata, args.in_dim_dict, args.hid_dim, args.out_dim,
                                          n_info=args.num_info, n_node=args.num_node,
                                          in_dim=args.in_dim, args=args)}
model_dict.update(cur_model_dict)
print(model_dict)




class ContrastiveWrapper(BasicModelWrapper):

    def common_step(self, batch: Batch, return_aug=False, return_hidden=False) -> Tuple[Tensor, Tensor]:
        batch_size = batch['patient'].batch_size
        x_dict = {"batch_size": batch_size}

        for n_type in batch.node_types:
            x_dict[n_type] = batch[n_type].bow_feat

        y = batch['patient'].y[:batch_size]
        y_hat = self.model(x_dict, batch.edge_index_dict, return_hidden=return_hidden)
        if return_hidden:
            y_hat = (y_hat[0][:batch_size], y_hat[1][:batch_size], y_hat[2][:batch_size], y_hat[3][:batch_size])
        else:
            y_hat = y_hat[:batch_size]

        return y_hat, y

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        y_hat, y = self.common_step(batch, return_hidden=True)
        y_hat, hid_pred, hid_assist, hid_aug = y_hat

        hid_neg = torch.roll(hid_aug, shifts=1)

        temperature = 0.1 if not hasattr(self.args, "temperature") else self.args.temperature
        info_loss_weight = 0.5 if not hasattr(self.args, "info_loss_weight") else self.args.info_loss_weight

        # infonce_loss = info_nce(y_hat, y_aug, y_neg, temperature=temperature)
        infonce_loss = info_nce(hid_pred, hid_aug, hid_neg, temperature=temperature)

        loss = (1 - info_loss_weight) * self.loss_fn(y_hat, y) + info_loss_weight * infonce_loss

        self.auc(y_hat, y.long())
        self.log(f'train_auc', self.auc, prog_bar=True, on_epoch=True)

        del y_hat, y
        return loss
