import pandas as pd
import pickle as pk

if __name__ == '__main__':
    for key in ["gender", "age"]:
        data_list = []
        for gt_1 in [True]:  # False
            save_file = f"{key}_{'upper' if gt_1 else 'lower'}_4.csv"
            data = pd.read_csv(save_file)
            print(f"{save_file} data size", data.shape)
            data_list.append(data)

        all_data = pd.concat(data_list)
        print("all_data size:", all_data.shape)

        # sort values
        all_data = all_data.sort_values(by=["drug_name", "SE_name"])
        save_file = f"{key}_ordered4.csv"
        all_data.to_csv(save_file)

        tmp = all_data.groupby(["drug_name", "SE_name"]).agg(
            {"count": 'sum', "ROR": "max", "ROR_upper": "count"}
        )
        tmp = tmp.rename(columns={"ROR": "max_ROR", "ROR_upper": f"related {key}s"})
        tmp = tmp.reset_index()
        save_file = f"{key}_grouped.csv"
        tmp.to_csv(save_file)

        print()
        print(f"{key} drugs：", len(tmp["drug_name"].unique()))
        print(f"{key} SEs：", len(tmp["SE_name"].unique()))
        print(f" {key} Grouped Data:", tmp.shape)

        SE_set = set(tmp["SE_name"].unique())
        pk.dump(SE_set, open(f"{key}_SE_set.pk", "wb"))

        tmp = all_data.groupby(["drug_name", f"{key}_id"]).agg({
            "SE_name": set,
            "count": 'sum',
            "ROR": "sum",
            "ROR_upper": "count"
        })
        tmp = tmp.reset_index()
        tmp = tmp.rename(columns={"ROR_upper": "SE_count"})
        save_file = f"{key}_stats.csv"
        tmp.to_csv(save_file)
