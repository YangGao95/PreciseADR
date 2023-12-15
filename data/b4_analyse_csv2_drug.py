import pandas as pd
import pickle as pk

if __name__ == '__main__':
    for key in ["drugs", "indications"]:
        data_list = []
        for gt_1 in [True, False]:  #
            save_file = f"{key}_{'upper' if gt_1 else 'lower'}_2.csv"
            data = pd.read_csv(save_file)
            print(f"{save_file} data size", data.shape)
            data_list.append(data)

        all_data = pd.concat(data_list)
        print("all_data size:", all_data.shape)

        # 重新排序
        all_data = all_data.sort_values(by=["drug_name", "SE_name"])
        save_file = f"{key}_ordered2.csv"
        all_data.to_csv(save_file)

        print(f"{key} nums：", len(all_data["drug_name"].unique()))
        print(f"{key} SEs：", len(all_data["SE_name"].unique()))

        tmp = all_data.groupby(["drug_name"]).agg({
            "SE_name": set,
            "ROR_upper": "count"
        })
        tmp = tmp.reset_index()
        tmp = tmp.rename(columns={"ROR_upper": "SE_count"})
        print(tmp)

        tmp = all_data.groupby(["SE_name", f"{key}_id"]).agg({
            "drug_name": set,
            "ROR_upper": "count"
        })
        tmp = tmp.reset_index()
        tmp = tmp.rename(columns={"ROR_upper": "drug_count"})
        print(tmp)

        SE_set = set(all_data["SE_name"].unique())
        pk.dump(SE_set, open(f"{key}_SE_set.pk", "wb"))



