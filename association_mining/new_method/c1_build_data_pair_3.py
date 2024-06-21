import pandas as pd

from data.b1_build_data import *
import os
from tqdm import tqdm
from collections import defaultdict
from rich import print


def load_origin_data():
    base_path = os.path.dirname(__file__)
    faers_root = f"{base_path}/../../data/Data/curated"
    target_file = f"{faers_root}/PLEASE-US-all.pk"
    all_pd_US_pro = pickle.load(open(target_file, "rb"))
    return all_pd_US_pro


def stat(all_pd_US_pro, save_name="PLEASE-US-all-drug-SE.csv"):
    # 初始化一个默认字典,用来存储统计结果
    result = defaultdict(lambda: [0, 0, 0, 0, 0])

    # 遍历DataFrame的每一行,进行统计
    for _, row in tqdm(all_pd_US_pro.iterrows(), total=len(all_pd_US_pro), desc="Processing rows"):
        gender = row['gender']
        age = row['age']
        drugs = row['drugs']
        drug_ses = row['SE']

        # 遍历每个药物及其副作用,更新统计结果
        for drug, drug_se in zip(drugs, drug_ses):
            key = f"{drug_se}_{drug}"

            if gender == '1':
                result[key][0] += 1  # male
            else:
                result[key][1] += 1  # female
            if age == "young":
                result[key][2] += 1  # young
            elif age == "adult":
                result[key][3] += 1  # adult
            else:
                result[key][4] += 1  # adult

    # 将统计结果转换为DataFrame
    result_df = pd.DataFrame.from_dict(result, orient='index', columns=["male", "female", "young", "adult", "elderly"])

    result_df["drug_SE"] = result_df.index
    result_df[["drug", "SE"]] = result_df["drug_SE"].str.split("_", expand=True)
    result_df = result_df[["drug_SE", "drug", "SE", "male", "female", "young", "adult", "elderly"]]
    result_df = result_df.reset_index(drop=True)
    result_df.index.name = None
    # 将新的DataFrame保存到文件
    result_df.to_csv(save_name, index=False)

    return result_df


if __name__ == '__main__':
    all_pd_US_pro = load_origin_data()
    all_pd_US_pro["age"] = all_pd_US_pro.apply(filter_age, axis=1)
    print(len(all_pd_US_pro))

    # statistics after filtering.
    # print("all_pd_US_pro data size:", len(all_pd_US_pro))
    # analyse_key2(all_pd_US_pro, key="gender", type="elem")
    # analyse_key2(all_pd_US_pro, key="age", type="elem")

    # new_df = stat(all_pd_US_pro)
    from scipy.stats import chi2_contingency

    new_df = pd.read_csv("PLEASE-US-all-drug-SE.csv")
    print(new_df.head())
    print(len(new_df))

    # 假设你的DataFrame名为'df'
    contingency_table = new_df[['male', 'female']]

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f'卡方统计量: {chi2:.2f}')
    print(f'p值: {p_value:.4f}')
    print(f'自由度: {dof}')

    # 找出显著相关的SE
    significant_SE = contingency_table.columns[p_value < 0.05]
    print(f'与性别显著相关的SE有: {", ".join(significant_SE)}')
