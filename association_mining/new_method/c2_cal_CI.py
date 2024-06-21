import pickle as pk
import numpy as np


def weird_division(n, d):
    return n / d if d else 0


def CI(ROR, A, B, C, D):
    ror = np.log(ROR)
    sq = 1.96 * np.sqrt(weird_division(1, A) + weird_division(1, B) + weird_division(1, C) + weird_division(1, D))
    CI_up = np.exp(ror + sq)
    CI_down = np.exp(ror - sq)
    return CI_up, CI_down


def get_CI_bound(key="gender"):

    drug_se_matrix, SE_map, drug_map, aim_map = pk.load(open(f"./drug_se_{key}_matrix_only_v4.pk", "rb"))
    filename = f"./se_drug_{key}_matrix_v4.pk"
    with open(filename, "rb") as file:
        SE_map, drug_map, aim_map, edge_index_0, edge_index_1, edge_index_2, gamma_ROR, sig, p_corrected = pk.load(
            file)
    gamma_ROR_CI_upper = np.ones_like(drug_se_matrix)
    gamma_ROR_CI_lower = np.ones_like(drug_se_matrix)
    for k in range(len(drug_map)):
        SE_aim_matrix = drug_se_matrix[k]
        for i in range(len(SE_map)):
            for j in range(len(aim_map)):
                a = SE_aim_matrix[i, j]

                if a == 0:
                    gamma_ROR[k][i][j] = 0

                else:
                    b = SE_aim_matrix[:, j].sum() - a
                    c = SE_aim_matrix[i, :].sum() - a
                    d = SE_aim_matrix.sum() - b - c - a

                    if b == 0 and c == 0:
                        pass
                    else:
                        ROR = gamma_ROR[k][i][j]
                        up, down = CI(ROR, a, b, c, d)
                        gamma_ROR_CI_upper[k][i][j] = up
                        gamma_ROR_CI_lower[k][i][j] = down

    with open(f"se_{key}_matrix_ci_v4.pk", "wb") as f:
        pk.dump([gamma_ROR, gamma_ROR_CI_upper, gamma_ROR_CI_lower], f)


if __name__ == '__main__':
    for key in ["gender", "age"]:
        get_CI_bound(key)
