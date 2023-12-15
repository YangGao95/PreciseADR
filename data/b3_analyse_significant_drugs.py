import pickle as pk
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd

if __name__ == '__main__':
    for key in ["drugs", "indications"]:
        for gt_1 in [True, False]:
            # get ROR
            filename = f"./se_{key}_matrix_v2.pk"
            threshold = 0
            with open(filename, "rb") as file:
                se_g_matrix, se_g_row, se_g_col, se_g_df, SE_map, drug_map, se_g_gamma_ROR, se_g_sig, se_g_p_corrected = pk.load(
                    file)

            SE_rev_map = {j: i for i, j in SE_map.items()}
            drug_rev_map = {j: i for i, j in drug_map.items()}

            # get ROR bounds
            filename = f"./se_{key}_matrix_ci_2.pk"
            with open(filename, "rb") as file:
                se_g_gamma_ROR, se_g_gamma_upper, se_g_gamma_ROR_lower = pk.load(file)

            edge_index = se_g_gamma_ROR.nonzero()
            row, col = edge_index[0][se_g_sig], edge_index[1][se_g_sig]
            exclude = []
            res = []
            row_set = set()

            n_drug = len(drug_map)

            # n_drug, n_gender/n_age
            sign_count = np.zeros((n_drug, len(se_g_gamma_ROR[0])))

            drug_SE_set = defaultdict(set)

            ror_res = []
            Item = namedtuple("Item", f"drug_name SE_name {key}_id count ROR ROR_lower ROR_upper")
            for i, j in zip(row, col):
                drug_id = j
                SE_id = i

                if gt_1 and se_g_gamma_ROR[i][j] > 1 and se_g_gamma_upper[i][j] > 1 and se_g_gamma_ROR_lower[i][j] > 1:
                    res.append((i, j))
                    row_set.add(i)
                    sign_count[drug_id][j] += 1 if j > 0 or key != "gender" else 0
                    drug_SE_set[drug_id].add(SE_id)

                    ror_res.append(
                        Item(
                            drug_rev_map[drug_id],
                            SE_rev_map[SE_id],
                            j,
                            se_g_matrix[i][j],
                            se_g_gamma_ROR[i][j],
                            se_g_gamma_upper[i][j],
                            se_g_gamma_ROR_lower[i][j]
                        )
                    )

                if not gt_1 and se_g_gamma_ROR[i][j] < 1 and se_g_gamma_upper[i][j] < 1 and se_g_gamma_ROR_lower[i][
                    j] < 1:
                    res.append((i, j))
                    row_set.add(i)
                    sign_count[drug_id][j] += 1 if j > 0 or key != "gender" else 0
                    drug_SE_set[drug_id].add(SE_id)

                    ror_res.append(
                        Item(
                            drug_rev_map[drug_id],
                            SE_rev_map[SE_id],
                            j,
                            se_g_matrix[i][j],
                            se_g_gamma_ROR[i][j],
                            se_g_gamma_upper[i][j],
                            se_g_gamma_ROR_lower[i][j]
                        )
                    )
                else:
                    if j == 0 and key == "gender":
                        pass
                    exclude.append((i, j))

            # print(res)
            print("res_len", len(res))

            # print(len(exclude))
            # print(len(row))

            save_file = f"{key}_{'upper' if gt_1 else 'lower'}_2.csv"
            pd.DataFrame(
                ror_res,
                columns=f"drug_name SE_name {key}_id count ROR ROR_upper ROR_lower".split()
            ).to_csv(save_file)
            print("Save to file:", save_file)

            SE_id_pair_list = [(SE_rev_map[each[0]], drug_rev_map[each[1]]) for each in res]
            SE_set = set()
            drug_set = set()
            for SE, drug in SE_id_pair_list:
                SE_set.add(SE)
                drug_set.add(drug)

            print(f"{key} num SE:{len(SE_set)}")
            print(f"{key} num {key}:{len(drug_set)}")
