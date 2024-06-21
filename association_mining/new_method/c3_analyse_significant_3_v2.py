import pickle as pk
import numpy as np
from collections import defaultdict, namedtuple
import pandas as pd


def build_final_pairs2(sign_count, mask):
    final_pairs = []
    for i, flag in zip(np.arange(len(sign_count)), mask):
        if flag:
            for drug_id, SE_id in res:
                if drug_id == i:
                    final_pairs.append((drug_rev_map[i], SE_rev_map[SE_id]))
    return final_pairs


if __name__ == '__main__':
    for key in ["gender", "age"]:
        for gt_1 in [True, False]:  #
            # get ROR values
            drug_se_matrix, SE_map, drug_map, aim_map = pk.load(open(f"drug_se_{key}_matrix_only_v4.pk", "rb"))
            filename = f"./se_drug_{key}_matrix_v4.pk"
            threshold = 0
            with open(filename, "rb") as file:
                SE_map, drug_map, aim_map, edge_index_0, edge_index_1, edge_index_2, gamma_ROR, sig, p_corrected = pk.load(
                    file)

            SE_rev_map = {j: i for i, j in SE_map.items()}
            drug_rev_map = {j: i for i, j in drug_map.items()}

            # build coo_matrix format matrix
            sig_matrix = np.zeros(drug_se_matrix.shape, dtype=np.bool_)
            for i, j, k, v in zip(edge_index_0, edge_index_1, edge_index_2, sig):
                sig_matrix[i][j][k] = v

            # get the upper and lower bound of ROR
            filename = f"./se_{key}_matrix_ci_v4.pk"
            with open(filename, "rb") as file:
                se_g_gamma_ROR, se_g_gamma_upper, se_g_gamma_ROR_lower = pk.load(file)

            # get index of non-zeros values
            edge_index = se_g_gamma_ROR.nonzero()

            sign_count = np.zeros((drug_se_matrix.shape[0], drug_se_matrix.shape[2]))

            drug_SE_set = defaultdict(set)

            res = []
            ror_res = []
            Item = namedtuple("Item", f"drug_name SE_name {key}_id count ROR ROR_lower ROR_upper")
            for drug_id, se_type_matrix in enumerate(drug_se_matrix):
                for SE_id, type_matrix in enumerate(se_type_matrix):
                    for type_id in range(len(type_matrix)):
                        if type_matrix.sum() < threshold or not sig_matrix[drug_id][SE_id][type_id]:
                            continue
                        if gt_1 and se_g_gamma_ROR[drug_id][SE_id][type_id] > 1 \
                                and se_g_gamma_upper[drug_id][SE_id][type_id] > 1 \
                                and se_g_gamma_ROR_lower[drug_id][SE_id][type_id] > 1:
                            res.append((drug_id, SE_id))
                            sign_count[drug_id][type_id] += 1
                            drug_SE_set[drug_id].add(SE_id)

                            ror_res.append(
                                Item(
                                    drug_rev_map[drug_id],
                                    SE_rev_map[SE_id],
                                    type_id,
                                    drug_se_matrix[drug_id][SE_id][type_id],
                                    se_g_gamma_ROR[drug_id][SE_id][type_id],
                                    se_g_gamma_upper[drug_id][SE_id][type_id],
                                    se_g_gamma_ROR_lower[drug_id][SE_id][type_id]
                                )
                            )

                        if not gt_1 and 0 < se_g_gamma_ROR[drug_id][SE_id][type_id] < 1 \
                                and 0 < se_g_gamma_upper[drug_id][SE_id][type_id] < 1 \
                                and 0 < se_g_gamma_ROR_lower[drug_id][SE_id][type_id] < 1:
                            res.append((drug_id, SE_id))
                            sign_count[drug_id][type_id] += 1
                            drug_SE_set[drug_id].add(SE_id)

                            ror_res.append(
                                Item(
                                    drug_rev_map[drug_id],
                                    SE_rev_map[SE_id],
                                    type_id,
                                    drug_se_matrix[drug_id][SE_id][type_id],
                                    se_g_gamma_ROR[drug_id][SE_id][type_id],
                                    se_g_gamma_upper[drug_id][SE_id][type_id],
                                    se_g_gamma_ROR_lower[drug_id][SE_id][type_id]
                                )
                            )
                        else:
                            pass

            # print(res)
            print(len(res))

            save_file = f"{key}_{'upper' if gt_1 else 'lower'}_4.csv"
            pd.DataFrame(
                ror_res,
                columns=f"drug_name SE_name {key}_id count ROR ROR_upper ROR_lower".split()
            ).to_csv(save_file)
            print("Save to file:", save_file)

            SE_set = set()
            drug_set = set()
            for SE, drug in res:
                SE_set.add(SE)
                drug_set.add(drug)

            print(f"{key} num SE:{len(SE_set)}")
            print(f"{key} num drug:{len(drug_set)}")

            print("final_pairs:", sign_count.sum())
            print()
