from data.build_dict import se_map as se_name_map, drug_map as drug_name_map
import pandas as pd
import numpy as np

if __name__ == '__main__':
    for key in ["gender", "age"]:  #
        filename = f"{key}_ordered2.csv"
        data_pd = pd.read_csv(filename)
        data_pd.replace([np.inf, -np.inf], np.nan, inplace=True)
        max_ROR = data_pd["ROR"].max()
        data_pd.replace(np.nan, max_ROR, inplace=True)
        data_pd["drug_name"] = data_pd.apply(lambda x: drug_name_map[x["drug_name"]], axis=1)
        data_pd["SE_name"] = data_pd.apply(lambda x: se_name_map[str(x["SE_name"])], axis=1)
        data_pd["type"] = data_pd.apply(lambda x: "1" if x["ROR"] > 1 else "2", axis=1)
        data_pd["ROR"] = data_pd.apply(lambda x: x["ROR"] if x["ROR"] > 1 else 1 / x["ROR"], axis=1)
        # data_pd["drug_name"] = data_pd.map(drug_name_map, axis=1)
        data_pd = data_pd[["drug_name", "SE_name", "ROR", "type", "count"]]
        data_pd.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_pd = data_pd.dropna()
        data_pd = data_pd.drop_duplicates(subset=['drug_name', 'SE_name'], keep="first")
        df_new = data_pd.rename(columns={"drug_name": "source", "SE_name": "target", "ROR": "weight"})
        df_new = df_new.sort_values(by="weight", ascending=False)
        df_new.to_csv(f"{filename[:-4]}_new.csv")
        print(df_new)

