import glob
import os
import os.path as osp
import pickle
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
from easydict import EasyDict as edict
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.functional import one_hot
from torch_geometric import transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

home_path = os.getenv("HOME")

base_path = osp.dirname(__file__)


class InfoMapping(object):
    num_elements = 1 + 3 + 30 + 13  # num_unknown + num_gender + num_weight + num_age

    def rev_map(self, i):
        if i == 0 or i == 1:  # 0 for unknown data, 1 for unknown gender
            return "unknown"
        elif i == 2:
            return "male"
        elif i == 3:
            return "female"
        elif 3 < i < (30 + 3 + 1):
            return f"weight:{10 * (i - 4)}"
        elif self.num_elements > i >= (30 + 3 + 1):
            return f"age:{10 * (i - 30 - 3 - 1)}"
        else:
            raise RuntimeError(f"Wrong value:{i}")

    def map_gender(self, gender):
        """
        map gender in records to index in feature
        :param gender:
        :return:
        """
        return int(gender) + 1

    def map_weight(self, weight):
        """
        map weight in records to index in feature
        weight larger than 300 is mapped to 0
        :param weight: weight (float) in records
        :return:
        """
        new_weight = int(float(weight))
        if new_weight > 300:
            new_weight = 0
        if new_weight < 0:
            new_weight = 0
        return new_weight // 10 + 3 + 1

    def map_age(self, age):
        """
        map age in records to index in feature
        age > 120 or age < 0  is mapped to -1
        :param age:
        :return:
        """
        new_age = int(float(age))
        if new_age > 120 or new_age < 0:
            new_age = -1
        return new_age // 10 + 1 + (3 + 30 + 1)


class PLEASESource(InMemoryDataset):

    def __init__(
            self,
            root: str = f"{base_path}/../data/Data/curated",
            n_data=0,
            se_type="all",
            force=False,
            raw_filename=None
    ):
        self.raw_filename = raw_filename
        self.se_type = se_type
        self.count = 0
        self.n_data = n_data
        self.root = root

        self.pre_init()

        if force:
            # remove existed files.
            filepath = os.path.join(root, "processed", self.processed_file_names[:-4] + "*")
            print(filepath)

            for f in glob.glob(filepath):
                os.remove(f)

        super(PLEASESource, self).__init__(root)
        self.post_init()

    def pre_init(self):
        """
        process before extracting from origin AE reports. Load data and build maps,
        :param root: file_path of AE reports
        :return:
        """
        all_data = self._load_data()
        if self.n_data == 0:
            self.n_data = len(all_data)

        # Feature index -> Feature name
        self.global_map = {}
        # personal infos
        self.info_mapping = InfoMapping()
        self.num_infos = self.info_mapping.num_elements

        # AE/SE map, Indication map, Drug map
        self.build_map(all_data)

        for i in range(self.info_mapping.num_elements):
            self.global_map[i] = self.info_mapping.rev_map(i)

        offset = self.info_mapping.num_elements
        for i, i_name in enumerate(self.i_map):
            self.global_map[i + offset] = i_name

        offset += len(self.i_map)
        for i, d_name in enumerate(self.d_map):
            self.global_map[i + offset] = d_name

    def post_init(self):
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.num_feat = self.data["patient"].bow_feat.size(1)
        self.num_se = self.data["patient"].y.size(1)

    @property
    def processed_file_names(self) -> str:
        return f'PLEASE_{self.se_type}_v1_{self.n_data}.pt'

    @property
    def raw_file_names(self) -> List[str]:
        if self.raw_filename is None:
            return [f'PLEASE-US-{self.se_type}.pk']
        else:
            return [self.raw_filename]

    def _build_person_graph(self, record):
        """
        transfer a record in FAERS to a subgraph
        :param record: An AE record
        :return:
        """
        if self.count % 100 == 0:
            print(f"{self.count}/{self.n_data}")

        infos = []
        attrs = []
        p_i_edge_index = [[], []]
        p_d_edge_index = [[], []]

        # 1. get infos from record
        indications = set(record["indications"])
        drugs = set(record["drugs"])
        se = set(record["SE"])

        # 2. get infos
        gender = record["gender"]
        age = record["age"]
        weight = record["weight"]
        infos.append(self.info_mapping.map_gender(gender))
        infos.append(self.info_mapping.map_weight(weight))
        infos.append(self.info_mapping.map_age(age))
        # todo add country ?
        infos.append(0)
        # country = record["country"]

        # 3. get edge_index
        for i in indications:
            p_i_edge_index[0].append(self.count)
            p_i_edge_index[1].append(self.i_map[i])

        for d in drugs:
            p_d_edge_index[0].append(self.count)
            p_d_edge_index[1].append(self.d_map[d])

        attrs.extend(infos[:-1])
        attrs.extend(np.array(p_i_edge_index[1]) + self.info_mapping.num_elements)
        attrs.extend(np.array(p_d_edge_index[1]) + self.info_mapping.num_elements + len(self.i_map))

        # origin SE id
        se_list = list(se)
        # new id
        new_se_list = []
        for each in se_list:
            new_se_list.append(self.se_map[each])

        self.count += 1
        return torch.LongTensor(infos).unsqueeze(0), torch.LongTensor(p_i_edge_index), \
            torch.LongTensor(p_d_edge_index), se_list, new_se_list, attrs

    def build_base_graph(self):
        """
        build the base graph, which consists of SE, Indication, Drug nodes.
        :return:
        """
        num_se = len(self.se_map)
        num_indication = len(self.i_map)
        num_drug = len(self.d_map)

        self.in_dim = self.info_mapping.num_elements + num_se + num_indication + num_drug

        self.hetero_graph["SE"].x = torch.arange(num_se)
        self.hetero_graph["indication"].x = torch.arange(num_indication)
        self.hetero_graph["drug"].x = torch.arange(num_drug)

        self.hetero_graph["indication"].bow_feat = one_hot(
            torch.arange(num_indication) + self.info_mapping.num_elements,  # Tensor + offset
            num_classes=self.in_dim).float().to_sparse()

        self.hetero_graph["drug"].bow_feat = one_hot(
            torch.arange(num_drug) + self.info_mapping.num_elements + num_indication,  # Tensor + offset
            num_classes=self.in_dim).float().to_sparse()

        self.hetero_graph["SE"].bow_feat = one_hot(
            torch.arange(num_se) + self.info_mapping.num_elements + num_indication + num_drug,  # Tensor + offset
            num_classes=self.in_dim).float().to_sparse()

    def _load_data(self):
        filename = f"{self.root}/{self.raw_file_names[0]}"
        assert osp.exists(filename), f"{filename} doesn't exist in path {self.root}."
        all_pd = pickle.load(open(filename, 'rb'))

        assert len(all_pd) > 0, "No data!!!"
        if self.n_data > 0:
            all_pd = all_pd.tail(self.n_data)
        else:
            self.n_data = len(all_pd)

        self.all_pd = all_pd
        return all_pd

    def process(self):
        all_pd = self.all_pd

        print("Generating Personal Graph START!!!!!")
        print("data size:", len(all_pd))

        self.hetero_graph = HeteroData()
        self.build_base_graph()
        self.extract_patient_graph(all_pd)

        torch.save(self.collate([self.hetero_graph]), self.processed_paths[0])

    def extract_patient_graph(self, all_pd):
        info_nodes_list = []
        se_name_index = []
        se_id_list = []
        all_p_i_edge_index = []
        all_p_d_edge_index = []
        all_y = []
        count = 0
        all_feats = []
        for info_nodes, p_i_edge_index, p_d_edge_index, se_name, se_id, attrs in tqdm(
                all_pd.apply(self._build_person_graph, axis=1)):
            info_nodes_list.append(info_nodes)
            all_p_i_edge_index.append(p_i_edge_index)
            all_p_d_edge_index.append(p_d_edge_index)
            se_name_index.append(se_name)  # old id
            se_id_list.append(se_id)  # new id
            y = torch.zeros(1, len(self.se_map))
            y[0, se_id] = 1
            all_y.append(y)
            all_feats.append(attrs)
            count += 1

        # BOW Feature
        all_x = []
        for attr in all_feats:
            x = torch.zeros([self.in_dim])
            x[attr] = 1
            all_x.append(x)
        x = torch.stack(all_x, dim=0)


        self.hetero_graph["patient"].x = torch.arange(count)
        self.hetero_graph["patient"].num_info = self.info_mapping.num_elements
        self.hetero_graph["patient", "p_i", "indication"].edge_index = torch.cat(all_p_i_edge_index, dim=1)
        self.hetero_graph["patient", "p_d", "drug"].edge_index = torch.cat(all_p_d_edge_index, dim=1)
        self.hetero_graph["patient"].y = torch.cat(all_y, dim=0).to_sparse()
        self.hetero_graph["patient"].ses = se_id_list
        self.hetero_graph["patient"].se_names = se_name_index

        self.hetero_graph["patient"].bow_feat = x.to_sparse()

    def build_map(self, all_pd):
        self.se_map = OrderedDict()
        self.i_map = OrderedDict()
        self.d_map = OrderedDict()

        save_path = f"{base_path}/{self.raw_file_names[0]}_map.pth"
        if os.path.exists(save_path):
            self.se_map, self.i_map, self.d_map = torch.load(save_path)

        else:
            def build_se(x):
                se = set(x["SE"])
                se = list(se)
                se.sort()
                for each in se:
                    if each not in self.se_map:
                        self.se_map[each] = len(self.se_map)

                return 1

            def build_indication_map(x):
                indications = set(x["indications"])
                indications = list(indications)
                indications.sort()
                for each in indications:
                    if each not in self.i_map:
                        self.i_map[each] = len(self.i_map)

                return 1

            def build_drug_map(x):
                drugs = list(set(x["drugs"]))
                drugs.sort()
                for each in drugs:
                    if each not in self.d_map:
                        self.d_map[each] = len(self.d_map)

                return 1

            all_pd.apply(build_se, axis=1)
            all_pd.apply(build_indication_map, axis=1)
            all_pd.apply(build_drug_map, axis=1)

        self.d_map_rev = {v: k for k, v in self.d_map.items()}
        self.i_map_rev = {v: k for k, v in self.i_map.items()}
        self.se_map_rev = {v: k for k, v in self.se_map.items()}


class DataModule(LightningDataModule):
    def __init__(
            self,
            n_data=10000,
            n_layer=2,
            batch_size=10240,
            add_SE=False,
            split="in_order",
            to_homo=False,
            filtered_SE=None,
            use_processed=True,
            args=None,
    ):
        super().__init__()
        self.use_processed = use_processed
        self.args = args

        if self.args is None:
            self.args = edict()

        if hasattr(args, "filtered_SE"):
            self.filtered_SE = args.filtered_SE
        else:
            self.filtered_SE = filtered_SE

        self.dataset = None
        self.setup_over = False
        self.num_neigh = args.num_neigh if hasattr(args, "num_neigh") else 10
        self.se_type = args.se_type if hasattr(args, "se_type") else "all"

        self.to_homo = to_homo
        self.n_data = n_data
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.add_SE = add_SE
        self.split = split
        self.num_train_per_class = self.args.num_train_per_class if hasattr(self.args, "num_train_per_class") else 10
        self.seed = self.args.seed if hasattr(self.args, "seed") else 123

        if not self.use_processed:
            data_path = self.get_final_filename()
            if os.path.exists(data_path):
                os.remove(data_path)
        self.load_processed()
        self.setup_over = True

    def load_processed(self):
        data_path = self.get_final_filename()

        if os.path.exists(data_path):
            self.data = torch.load(data_path)
        else:
            self.setup()
            if self.to_homo:
                self.data.bow_feat = self.data.bow_feat.to_sparse()
                self.data.y = self.data.y.to_sparse()
            else:
                for nt in self.data.node_types:
                    self.data[nt].bow_feat = self.data[nt].bow_feat.to_sparse()
                self.data["patient"].y = self.data["patient"].y.to_sparse()
            torch.save(self.data, data_path)

        if self.to_homo:
            self.data.bow_feat = self.data.bow_feat.to_dense()
            self.data.y = self.data.y.to_dense()
        else:
            n_feat = self.data["indication"].bow_feat.size(1)
            n_i = self.data["indication"].bow_feat.size(0)
            n_d = self.data["drug"].bow_feat.size(0)
            n_se = self.data["SE"].bow_feat.size(0)

            self.i_start = n_feat - n_se - n_d - n_i
            self.i_end = n_feat - n_se - n_d
            self.d_start = n_feat - n_se - n_d
            self.d_end = n_feat - n_se

            for nt in self.data.node_types:
                self.data[nt].bow_feat = self.data[nt].bow_feat.to_dense()
            self.data["patient"].y = self.data["patient"].y.to_dense()

        del self.data["patient"].ses
        del self.data["patient"].se_names

    def get_final_filename(self):
        prefix = "homo" if self.to_homo else "hetero"
        data_path = f"{base_path}/{prefix}_{self.se_type}_{self.n_data}_{self.split}_" \
                    f"{self.num_train_per_class}_{self.seed}_add_se_{self.add_SE}.pth"
        print("load data from:", data_path)
        return data_path

    def to_hetero_data(self):
        assert self.data is not None
        num_info = self.data["patient"].num_info
        self.node_types, self.edge_types = self.data.metadata()
        y = self.data['patient'].y.clone()
        self.n_se = y.size(1)
        self.data["patient"].num_info = num_info
        return self.data

    def to_homo_data(self):
        assert self.data is not None
        num_info = self.data["patient"].num_info
        node_types, edge_types = self.data.metadata()
        y = self.data['patient'].y.clone()
        n_se = y.size(1)
        old_bow_size = self.data["patient"].bow_feat.size(1)
        for nt in node_types:
            data_size = self.data[nt].x.size(0)
            if nt != "patient":
                self.data[nt].y = torch.zeros([data_size, n_se])
                self.data[nt].train_mask = torch.zeros([data_size], dtype=torch.bool)
                self.data[nt].val_mask = torch.zeros([data_size], dtype=torch.bool)
                self.data[nt].test_mask = torch.zeros([data_size], dtype=torch.bool)
            if nt != "SE":
                self.data[nt].bow_feat = torch.cat([self.data[nt].bow_feat, torch.zeros([data_size, n_se])], dim=-1)

        n_se = self.data["SE"].bow_feat.size(0)
        self.data["SE"].bow_feat = torch.cat([torch.zeros([n_se, old_bow_size]), torch.eye(n_se)], dim=-1)
        print(self.data)
        homo_data = self.data.to_homogeneous()
        homo_data.num_info = num_info
        return homo_data

    def save_labels(self):
        train_mask, val_mask, test_mask \
            = self.data["patient"].train_mask, self.data["patient"].val_mask, self.data["patient"].test_mask
        f_name = f"{self.split}_{self.n_data}_label_mask.pth"
        torch.save([train_mask, val_mask, test_mask], f_name)

    def get_data_info(self):
        self.i_start = self.dataset.info_mapping.num_elements
        self.i_end = self.i_start + len(self.dataset.i_map)
        self.d_start = self.i_end
        self.d_end = self.i_end + len(self.dataset.d_map)
        self.dataset.data["patient"].y = self.dataset.data["patient"].y.to_dense()
        for n_type in self.dataset.data.node_types:
            self.dataset.data[n_type].bow_feat = self.dataset.data[n_type].bow_feat.to_dense()
        self.in_dim = self.dataset.data["patient"].bow_feat.size(1)

    def load_data(self):
        if self.dataset is None:
            self.dataset = PLEASESource(n_data=self.n_data, se_type=self.se_type, force=not self.use_processed)
            self.get_data_info()
        return self.dataset

    def prepare_data(self):
        return self.load_data()

    def setup(self, stage: Optional[str] = None):
        if not self.setup_over:
            # 1. load data
            self.dataset = self.load_data()
            self.data = self.dataset.data

            # 2. label split
            if self.split == "in_order":
                self.data = FaersRandomNodeSplit()(self.data)
            elif self.split == "by_label":
                self.data = FaersNodeSplitByLabel(num_train_per_class=self.num_train_per_class)(self.data)
            else:
                self.data = T.RandomNodeSplit(num_val=0.125, num_test=0.125)(self.data)
            # 3. Add SE nodes
            if self.add_SE:
                self.data = AddSEEdges()(self.data)
            # 4. Add reverse links
            self.data = T.ToUndirected(merge=False)(self.data)

            if self.to_homo:
                self.data = self.to_homo_data()
            else:
                self.data = self.to_hetero_data()

            if not self.to_homo:
                print(self.data.metadata)
                self.metadata = self.data.metadata

            print("Validation:", self.data.validate())

            self.setup_over = True

    def dataloader(self, mask: Tensor, shuffle: bool, num_workers: int = 8, mode="train"):
        batch_size = self.batch_size
        if self.to_homo:

            dataloader = NeighborLoader(self.data, num_neighbors=[self.num_neigh] * (self.n_layer),
                                        input_nodes=mask, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers,
                                        persistent_workers=num_workers > 0)
        else:
            dataloader = NeighborLoader(self.data, num_neighbors=[self.num_neigh] * (self.n_layer),
                                        input_nodes=('patient', mask), batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers,
                                        persistent_workers=num_workers > 0)
            # dataloader.batch_size = batch_size
            dataloader.input_nodes = ('patient', mask)

        dataloader.num_neighbors = [self.num_neigh] * (self.n_layer)
        return dataloader

    def train_dataloader(self):
        if self.to_homo:
            return self.dataloader(self.data.train_mask, shuffle=True)
        else:
            return self.dataloader(self.data['patient'].train_mask, shuffle=True)

    def val_dataloader(self):
        if self.to_homo:
            return self.dataloader(self.data.val_mask, shuffle=False)
        else:
            return self.dataloader(self.data['patient'].val_mask, shuffle=False)

    def test_dataloader(self):
        if self.to_homo:
            return self.dataloader(self.data.test_mask, shuffle=False)
        else:
            return self.dataloader(self.data['patient'].test_mask, shuffle=False)


class FaersRandomNodeSplit(BaseTransform):
    def __init__(self,
                 split="in_order",
                 num_val=0.25,
                 num_test=0.25,
                 merge_seen_and_train=True
                 ):
        self.merge_seen_and_train = merge_seen_and_train
        self.num_val = num_val
        self.num_test = num_test
        self.split = split
        super(FaersRandomNodeSplit, self).__init__()

    def __call__(
            self,
            data,
    ):
        # only faers nodes has label
        num_faers = data["patient"].x.size(0)
        num_seen = num_faers // 2

        seen_mask = torch.zeros(num_faers, dtype=torch.bool)
        train_mask = torch.zeros(num_faers, dtype=torch.bool)
        val_mask = torch.zeros(num_faers, dtype=torch.bool)
        test_mask = torch.zeros(num_faers, dtype=torch.bool)

        if isinstance(self.num_val, float):
            num_val = round((num_faers - num_seen) * self.num_val)
        else:
            num_val = self.num_val

        if isinstance(self.num_test, float):
            num_test = round((num_faers - num_seen) * self.num_test)
        else:
            num_test = self.num_test

        assert num_test < num_faers
        assert num_val < num_faers

        if self.split == "in_order":
            test_mask[-num_test:] = True
            val_mask[-num_test - num_val:-num_test] = True
            train_mask[-(num_faers - num_seen):-num_test - num_val] = True
            seen_mask[-num_faers:-(num_faers - num_seen)] = True
        else:
            perm = torch.randperm(num_faers)
            val_mask[perm[:num_val]] = True
            test_mask[perm[num_val:num_val + num_test]] = True
            train_mask[perm[num_val + num_test:-num_seen]] = True
            seen_mask[perm[-num_seen:]] = True

        if self.merge_seen_and_train:
            train_mask = torch.logical_or(train_mask, seen_mask)

        data["patient"].train_mask, data["patient"].val_mask, data[
            "patient"].test_mask = train_mask, val_mask, test_mask
        data["patient"].seen_mask = seen_mask
        return data


class FaersNodeSplitByLabel(FaersRandomNodeSplit):
    """
     num_train_per_class for train
    25% for validation
    rest for test
    """

    def __init__(self,
                 split="in_order",
                 num_train_per_class=10,
                 num_val=0.25
                 ):
        self.split = split
        self.num_val = num_val
        self.num_train_per_class = num_train_per_class
        super(FaersNodeSplitByLabel, self).__init__()

    def __call__(
            self,
            data,
    ):
        # only faers nodes has label
        num_faers = data["patient"].x.size(0)

        train_mask = torch.zeros(num_faers, dtype=torch.bool)
        val_mask = torch.zeros(num_faers, dtype=torch.bool)
        test_mask = torch.zeros(num_faers, dtype=torch.bool)

        if isinstance(self.num_val, float):
            num_val = round(num_faers * self.num_val)
        else:
            num_val = self.num_val

        assert num_val < num_faers

        y = data["patient"].y
        num_classes = y.size(1)

        for c in range(num_classes):
            idx = (y[:, c] == 1).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))]
            idx = idx[:self.num_train_per_class]
            train_mask[idx] = True

        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        val_mask[remaining[:num_val]] = True
        test_mask[remaining[num_val:]] = True

        data["patient"].train_mask, data["patient"].val_mask, data[
            "patient"].test_mask = train_mask, val_mask, test_mask
        data["patient"].seen_mask = train_mask
        return data


class AddSEEdges(BaseTransform):
    def __call__(
            self,
            data,
    ):
        if ("patient", "p_se", "SE") in data:
            return data
        # mask = data["patient"].seen_mask if hasattr(data["patient"], "seen_mask") else data["patient"].train_mask
        mask = data["patient"].train_mask

        new_edges_index = []
        aim_nodes = data["patient"].x[mask].tolist()

        for node in aim_nodes:
            # dst = data["patient"].ses[node]
            dst = data["patient"].y[node]
            dst = (torch.arange(len(dst))[dst.ge(1)]).tolist()

            src = [node] * len(dst)
            new_edges_index.append(
                torch.stack([torch.LongTensor(src), torch.LongTensor(dst)])
            )
        data["patient", "p_se", "SE"].edge_index = torch.cat(new_edges_index, dim=1)
        return data


if __name__ == '__main__':
    dataset = PLEASESource(n_data=0, force=True)
    DataModule(n_data=0, use_processed=False)
