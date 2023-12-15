# %%
# load data
import os.path
import pickle as pk
import pickle
import numpy as np

from data.meanless_se import drop_list
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import pandas as pd
import os

base_path = os.path.dirname(__file__)

def load_origin_data():
    faers_root = f"{base_path}/Data/curated"
    target_file = f"{faers_root}/filtered_data_v1.pk"
    all_pd_US_pro = pickle.load(open(target_file, "rb"))
    return all_pd_US_pro


def build_SE_map(all_data_pd):
    se_list = []

    def scan_se(x):
        se_list.extend(x["SE"])

    all_data_pd.apply(scan_se, axis=1)

    se_name, se_count = np.unique(se_list, return_counts=True)

    # drop_list = build_drop_SE_list()
    # %%
    SE_list, SE_count = [], []
    for name, count in zip(se_name, se_count):
        if name not in drop_list and count > 100:
            SE_list.append(name)
            SE_count.append(count)

    return {se_name: i for i, se_name in enumerate(SE_list)}


def build_drug_map(all_data_pd, key="drugs", type="list"):
    drug_list = []

    def scan_drug(x):
        if type == "list":
            drug_list.extend(x[key])
        else:
            drug_list.append(x[key])

    all_data_pd.apply(scan_drug, axis=1)
    drug_set = set(drug_list)
    drug_map = {drug: i for drug, i in zip(drug_set, range(len(drug_set)))}
    return drug_map


def stat_matrix(SE_map, drug_map, all_pd_US_pro, key="drugs", type="list"):
    se_drug_matrix = np.zeros((len(SE_map), len(drug_map)))

    def scan_se_drug(x):
        for SE in set(x["SE"]):
            if SE not in SE_map:
                continue
            SE_index = SE_map[SE]

            if type == "list":
                for d in set(x[key]):
                    d_index = drug_map[d]
                    se_drug_matrix[SE_index][d_index] += 1
            else:
                d = x[key]
                d_index = drug_map[d]
                se_drug_matrix[SE_index][d_index] += 1

    all_pd_US_pro.apply(scan_se_drug, axis=1)
    return se_drug_matrix


def stat_p_value(SE_map, drug_map, se_drug_matrix):
    gamma_ROR = np.zeros((len(SE_map), len(drug_map)))
    gamma_p_value = np.zeros((len(SE_map), len(drug_map)))

    gamma_ROR_CI_upper = np.zeros((len(SE_map), len(drug_map)))
    gamma_ROR_CI_lower = np.zeros((len(SE_map), len(drug_map)))

    for i in range(len(SE_map)):
        for j in range(len(drug_map)):
            a = se_drug_matrix[i, j]

            if a == 0:
                gamma_ROR[i][j] = 0
                gamma_p_value[i][j] = 1

            else:
                b = se_drug_matrix[:, j].sum() - a
                c = se_drug_matrix[i, :].sum() - a
                # d = se_drug_matrix[:, :].sum() - se_drug_matrix[:, j].sum() - se_drug_matrix[i, :].sum() + a
                d = se_drug_matrix[:, :].sum() - b - c - a
                gamma_ROR[i][j], gamma_p_value[i][j] = stats.fisher_exact([[a, b], [c, d]])

    print(gamma_ROR)
    # %%
    # multipletests
    edge_index = gamma_ROR.nonzero()
    sig, p_corrected = multipletests(pvals=gamma_p_value[edge_index], alpha=0.05, method='bonferroni')[0:2]

    return gamma_ROR, edge_index, sig, p_corrected


def analyse(SE_map, drug_map, edge_index, sig, p_corrected, se_drug_matrix, key="drug"):
    SE_map_rev = {v: k for k, v in SE_map.items()}
    drug_map_rev = {v: k for k, v in drug_map.items()}
    id_index = np.argsort(p_corrected[sig])
    row = edge_index[0][sig][id_index]
    col = edge_index[1][sig][id_index]
    print(row, col)
    res_list = []
    for i, j in zip(row, col):
        res = {}
        res["SE"] = SE_map_rev[i]
        res[f"{key}"] = drug_map_rev[j]

        res["count"] = se_drug_matrix[i][j]
        res["SE_count"] = se_drug_matrix[i, :].sum()
        res[f"{key}_count"] = se_drug_matrix[:, j].sum()

        res[f"count/{key}_count"] = res["count"] / res[f"{key}_count"]
        res["count/SE_count"] = res["count"] / res["SE_count"]

        mean_row = se_drug_matrix[i, :].sum() / len(se_drug_matrix[i, :].nonzero()[0])
        res["SE_mean"] = mean_row
        res["SE_median"] = np.median(se_drug_matrix[i, se_drug_matrix[i, :].nonzero()[0]])

        mean_col = se_drug_matrix[:, j].sum() / len(se_drug_matrix[:, j].nonzero()[0])
        res[f"{key}_mean"] = se_drug_matrix[i][j] / mean_row
        res[f"{key}_median"] = np.median(se_drug_matrix[se_drug_matrix[:, j].nonzero()[0], j])

        res_list.append(res)

    d_df = pd.DataFrame(res_list)
    return d_df


def analyse_key(all_pd_US_pro, key="drugs", type="list"):
    SE_map = build_SE_map(all_pd_US_pro)
    drug_map = build_drug_map(all_pd_US_pro, key=key, type=type)

    print("SE len", len(SE_map))
    print(f"{key} len", len(drug_map))

    se_drug_matrix = stat_matrix(SE_map, drug_map, all_pd_US_pro, key=key, type=type)
    gamma_ROR, edge_index, sig, p_corrected = stat_p_value(SE_map, drug_map, se_drug_matrix)
    d_df = analyse(SE_map, drug_map, edge_index, sig, p_corrected, se_drug_matrix)
    with open(f"{base_path}/se_{key}_matrix_v2.pk", "wb") as f:
        pk.dump([se_drug_matrix, edge_index[0], edge_index[1], d_df, SE_map, drug_map, gamma_ROR, sig, p_corrected], f)


# 修改weight 值
def filter_weight(x):
    i = float(x.weight)
    if i <= 50:
        return "weight<=50"
    elif 50 < i <= 100:
        return "50<weight<= 100"
    elif 100 < i <= 150:
        return "100<weight<= 150"
    else:
        return "weight > 150"


def filter_age(x):
    i = x.age
    if int(0) <= int(i) < int(20):
        return "young"
    elif int(20) < int(i) < int(65):
        return "adult"
    else:
        return "elderly"


if __name__ == '__main__':
    all_pd_US_pro = load_origin_data()
    print(len(all_pd_US_pro))

    all_pd_US_pro["weight"] = all_pd_US_pro.apply(filter_weight, axis=1)
    all_pd_US_pro["age"] = all_pd_US_pro.apply(filter_age, axis=1)

    # 初步筛选后的统计信息
    print("all_pd_US_pro data size:", len(all_pd_US_pro))

    analyse_key(all_pd_US_pro, key="indications", type="list")
    analyse_key(all_pd_US_pro, key="drugs", type="list")

