import pickle as pk
import numpy as np

import os

base_path = os.path.dirname(__file__)

def weird_division(n, d):
    return n / d if d else 0


def CI(ROR, A, B, C, D):
    ror = np.log(ROR)
    sq = 1.96*np.sqrt(weird_division(1, A) + weird_division(1, B) + weird_division(1, C) +weird_division(1, D))
    CI_up = np.exp(ror + sq)
    CI_down = np.exp(ror - sq)
    return CI_up, CI_down


def get_CI_bound(key="gender"):
    filename = f"./se_{key}_matrix_v2.pk"
    threshold = 1000
    threshold = 0
    with open(filename, "rb") as file:
        se_g_matrix, se_g_row, se_g_col, se_g_df, SE_map, gender_map, se_g_gamma_ROR, se_g_sig, se_g_p_corrected = pk.load(
            file)
    gamma_ROR_CI_upper = np.ones_like(se_g_matrix)
    gamma_ROR_CI_lower = np.ones_like(se_g_matrix)
    for i in range(len(se_g_matrix)):
        counts = se_g_matrix[i].sum()
        if counts < threshold:
            continue

        for j in range(len(se_g_matrix[i])):
            a = se_g_matrix[i, j]

            if a == 0:
                continue
            else:
                b = se_g_matrix[:, j].sum() - a
                c = se_g_matrix[i, :].sum() - a
                d = se_g_matrix[:, :].sum() - se_g_matrix[:, j].sum() - se_g_matrix[i, :].sum() + a
                ROR = se_g_gamma_ROR[i][j]
                up, down = CI(ROR, a, b, c, d)
                gamma_ROR_CI_upper[i][j] = up
                gamma_ROR_CI_lower[i][j] = down
    with open(f"se_{key}_matrix_ci_2.pk", "wb") as f:
        pk.dump([se_g_gamma_ROR, gamma_ROR_CI_upper, gamma_ROR_CI_lower], f)


if __name__ == '__main__':
    for key in ["drugs", "indications"]:
        get_CI_bound(key)