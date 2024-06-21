import pandas as pd
import torch

aehgn_list, improve = torch.load("../heatmap/AEHGN_best_se_set_v3.pth")
# 读取两个CSV文件
file1 = pd.read_csv("age_AUC_ge_0.csv")
file2 = pd.read_csv("gender_AUC_ge_0.csv")
by = "base"
# by = "diff"
# 根据"diff"列从大到小排序
file1 = file1.sort_values(by=by, ascending=False)
file2 = file2.sort_values(by=by, ascending=False)

file1["SE_id"] = file1["SE_id"].astype(str)
file2["SE_id"] = file2["SE_id"].astype(str)
file1 = file1[file1["SE_id"].isin(aehgn_list)]
file2 = file2[file2["SE_id"].isin(aehgn_list)]
# 初始化两个列表用于存储最终结果
result_file1 = []
result_file2 = []

# 设置要挑选的TOP N条数据
N = 40

# 用于记录已选择的SE_id
selected_se_ids = set()

# 挑选数据，尽可能平均分配
for i in range(N):
    if i % 2 == 1:
        # 从file1挑选数据
        for _, row in file1.iterrows():
            if row["SE_id"] not in selected_se_ids:
                result_file1.append(row)
                selected_se_ids.add(row["SE_id"])
                break
    else:
        # 从file2挑选数据
        for _, row in file2.iterrows():
            if row["SE_id"] not in selected_se_ids:
                result_file2.append(row)
                selected_se_ids.add(row["SE_id"])
                break

# 将结果转换为DataFrame
result_file1 = pd.DataFrame(result_file1)
result_file2 = pd.DataFrame(result_file2)

# 打印结果或保存到新的CSV文件
print("Result from File 1:")
print(result_file1)

print("Result from File 2:")
print(result_file2)

# 如果需要将结果保存到新的CSV文件，可以使用to_csv方法
result_file1.to_csv("result_file1_age.csv", index=False)
result_file2.to_csv("result_file2_gender.csv", index=False)
