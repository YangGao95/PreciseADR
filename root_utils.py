import torch
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cal_auc_sperately(y_pred, y, top_k=1):
    num_class = y.size(1)
    auc_res = []
    p_res = []
    r_res = []
    # print(y_pred.shape)
    # print(y_pred)
    # print(y)

    top_index = torch.argsort(y_pred, dim=1, descending=True)
    y_pred_label = torch.zeros_like(y_pred, dtype=torch.long)
    for i in range(len(top_index)):
        index = top_index[i][:top_k]
        y_pred_label[i][index] = 1

    # y_pred_label = torch.nn.functional.one_hot(torch.argmax(y_pred, dim=1), num_classes=num_class)
    y_pred_logits = y_pred * y_pred_label

    for i in range(num_class):
        label = y[:, i]
        pred_label = y_pred_label[:, i]
        pred_logits = y_pred_logits[:, i]
        if label.sum() == label.size(0) or label.sum() == 0:
            auc_res.append(-1)
            p_res.append(-1)
            r_res.append(-1)
        else:
            auc_res.append(roc_auc_score(label, pred_logits))
            p_res.append(precision_score(label, pred_label))
            r_res.append(recall_score(label, pred_label))

    return auc_res, p_res, r_res


def cal_sensitivity_and_specificity(y_pred, y, top_k=20):
    num_class = y.size(1)
    sen_res = []
    spe_res = []

    top_index = torch.argsort(y_pred, dim=1, descending=True)
    y_pred_label = torch.zeros_like(y_pred, dtype=torch.long)
    for i in range(len(top_index)):
        index = top_index[i][:top_k]
        y_pred_label[i][index] = 1

    for i in range(num_class):
        label = y[:, i]
        pred = y_pred_label[:, i]
        sen_mask = (label == 1)
        spe_mask = (label == 0)

        sen_res.append(accuracy_score(label[sen_mask], pred[sen_mask]))
        spe_res.append(accuracy_score(label[spe_mask], pred[spe_mask]))

    return sen_res, spe_res


def build_save_path(args, score=0.0):
    return f"{args.dataset}_{args.model_name}_seed_{args.seed}_n_gnn_{args.n_gnn}_n_mlp:{args.n_mlp}_score:{score:.4f}.ckpt"
