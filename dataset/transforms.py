import torch
from torch_geometric.transforms import BaseTransform


class AddSEEdges(BaseTransform):
    """
    Add patient-> Side effects edges
    """

    def __call__(
            self,
            data,
    ):
        if ("patient", "p_se", "SE") in data:
            return data

        mask = data["patient"].train_mask

        new_edges_index = []
        aim_nodes = data["patient"].x[mask].tolist()

        for node in aim_nodes:
            dst = data["patient"].y[node]
            dst = (torch.arange(len(dst))[dst.ge(1)]).tolist()

            src = [node] * len(dst)
            new_edges_index.append(
                torch.stack([torch.LongTensor(src), torch.LongTensor(dst)])
            )
        data["patient", "p_se", "SE"].edge_index = torch.cat(new_edges_index, dim=1)
        return data


class FaersRandomNodeSplit(BaseTransform):
    def __init__(self,
                 split="in_order",
                 num_val=0.125,
                 num_test=0.125,
                 ):
        self.num_val = num_val
        self.num_test = num_test
        self.split = split
        super(FaersRandomNodeSplit, self).__init__()

    def __call__(
            self,
            data,
    ):
        # only patient nodes has label
        num_all = data["patient"].x.size(0)

        train_mask = torch.zeros(num_all, dtype=torch.bool)
        val_mask = torch.zeros(num_all, dtype=torch.bool)
        test_mask = torch.zeros(num_all, dtype=torch.bool)

        num_val = round(num_all * self.num_val) if 0 < self.num_val < 1 else self.num_val
        num_test = round(num_all * self.num_test) if 0 < self.num_val < 1 else self.num_test

        assert num_test < num_all
        assert num_val < num_all

        if self.split == "in_order":
            test_mask[-num_test:] = True
            val_mask[-num_test - num_val:-num_test] = True
            train_mask[:-num_test - num_val] = True
        else:
            perm = torch.randperm(num_all)
            val_mask[perm[:num_val]] = True
            test_mask[perm[num_val:num_val + num_test]] = True
            train_mask[perm[num_val + num_test:]] = True

        data["patient"].train_mask, data["patient"].val_mask, data[
            "patient"].test_mask = train_mask, val_mask, test_mask
        return data


class FaersNodeSplitByTime(BaseTransform):
    def __init__(self,
                 year_val="2021",
                 year_test="2022",
                 ):
        self.year_val = year_val
        self.year_test = year_test
        super(FaersNodeSplitByTime, self).__init__()

    def __call__(
            self,
            data,
    ):
        # only patient nodes has label
        date = data["patient"].date
        val_mask = torch.BoolTensor([self.year_val in each for each in date])
        test_mask = torch.BoolTensor([self.year_test in each for each in date])
        assert val_mask.sum() > 0
        assert test_mask.sum() > 0
        train_mask = ~ (torch.logical_or(val_mask, test_mask))

        data["patient"].train_mask, data["patient"].val_mask, data[
            "patient"].test_mask = train_mask, val_mask, test_mask
        return data


class FaersNodeSplitByRegion(BaseTransform):
    def __init__(self,
                 train_region="US",
                 val_rate=0.25,
                 ):
        self.train_region = train_region
        self.val_rate = val_rate
        super(FaersNodeSplitByRegion, self).__init__()

    def __call__(
            self,
            data,
    ):
        # only patient nodes has label
        country = data["patient"].country
        train_mask = torch.BoolTensor([self.train_region == each for each in country])
        assert train_mask.sum() > 0, f"No {self.train_region} data"
        test_mask = ~ train_mask
        assert test_mask.sum() > 0, f"Only {self.train_region} data"

        num_train = train_mask.sum().item()
        num_val = round(num_train * self.val_rate)

        train_indices = torch.nonzero(train_mask).view(-1)
        val_indices = train_indices[:num_val]
        train_indices = train_indices[num_val:]

        train_mask = torch.zeros_like(train_mask)
        train_mask[train_indices] = True

        val_mask = torch.zeros_like(train_mask)
        val_mask[val_indices] = True



        data["patient"].train_mask, data["patient"].val_mask, data[
            "patient"].test_mask = train_mask, val_mask, test_mask
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
