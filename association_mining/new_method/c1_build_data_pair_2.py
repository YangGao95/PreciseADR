from association_mining.c1_build_data_pair import *

def stat_p_value2(SE_map, drug_map, aim_map, drug_se_matrix):
    gamma_ROR = np.ones(
        (len(drug_map), len(SE_map), len(aim_map))
    )
    gamma_p_value = np.ones(
        (len(drug_map), len(SE_map), len(aim_map))
    )
    for k in range(len(drug_map)):
        SE_aim_matrix = drug_se_matrix[k]
        for i in range(len(SE_map)):
            for j in range(len(aim_map)):
                a = SE_aim_matrix[i, j]

                if a == 0:
                    gamma_ROR[k][i][j] = 1
                    gamma_p_value[k][i][j] = 1

                else:
                    b = SE_aim_matrix[i, :].sum() - a  # 同SE，不同性别
                    c = SE_aim_matrix[:, j].sum() - a  # 同性别，不同SE

                    d = SE_aim_matrix.sum() - b - c - a  # 不同SE，不同性别

                    if (b == 0 or c == 0 or d == 0) or (a+b+c < 100):
                        gamma_ROR[k][i][j] = 1
                        gamma_p_value[k][i][j] = 1
                        pass
                    else:
                        gamma_ROR[k][i][j], gamma_p_value[k][i][j], _, _ = stats.chi2_contingency([[a, b], [c, d]])

    print(gamma_ROR)

    print(gamma_ROR)
    # %%
    # multipletests
    edge_index = gamma_ROR.nonzero()
    sig, p_corrected = multipletests(pvals=gamma_p_value[edge_index], alpha=0.05, method='bonferroni')[0:2]

    return gamma_ROR, edge_index, sig, p_corrected


def analyse2(SE_map, drug_map, aim_map, edge_index, sig, p_corrected, se_drug_matrix, key="drug"):
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


def analyse_key3(all_pd_US_pro, key="drugs", type="list", force=True):
    # Build a global map, name(str) -> index(int).
    SE_map = build_SE_map(all_pd_US_pro)
    drug_map = build_drug_map(all_pd_US_pro, key="drugs", type="list")
    aim_map = build_drug_map(all_pd_US_pro, key=key, type=type)

    print("SE len", len(SE_map))
    print(f"drugs len", len(drug_map))

    save_path = f"drug_se_{key}_matrix_only_v4.pk"
    if os.path.exists(save_path) and not force:
        drug_se_matrix, SE_map, drug_map, aim_map = pk.load(open(save_path, "rb"))
    else:
        drug_se_matrix = stat_matrix2(SE_map, drug_map, aim_map, all_pd_US_pro, key=key, type=type)
        with open(save_path, "wb") as f:
            pk.dump([drug_se_matrix, SE_map, drug_map, aim_map], f)

    # calculate odds ratio and p_value
    gamma_ROR, edge_index, sig, p_corrected = stat_p_value2(SE_map, drug_map, aim_map, drug_se_matrix)
    with open(f"se_drug_{key}_matrix_v4.pk", "wb") as f:
        pk.dump([SE_map, drug_map, aim_map, edge_index[0], edge_index[1], edge_index[2], gamma_ROR, sig, p_corrected],
                f)


if __name__ == '__main__':
    all_pd_US_pro = load_origin_data()
    all_pd_US_pro["age"] = all_pd_US_pro.apply(filter_age, axis=1)
    print(len(all_pd_US_pro))

    # statistics after filtering.
    print("all_pd_US_pro data size:", len(all_pd_US_pro))
    analyse_key3(all_pd_US_pro, key="gender", type="elem")
    analyse_key3(all_pd_US_pro, key="age", type="elem")
