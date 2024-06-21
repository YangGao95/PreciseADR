from analyse.heatmap.a2_get_AEHGN_best import *
from dataset.please import PLEASESource
import pandas as pd
from data.build_dict import se_map as se_name_map, drug_map as drug_name_map
base_path = os.path.dirname(__file__)
info = {
    "gender": f"{base_path}/../../gender_HGT_seed_42_n_gnn_3_n_mlp:3_res_False_use_jk:Truescore:0.8233.ckpt.res",
    "age": f"{base_path}/../../age_HGT_seed_42_n_gnn_3_n_mlp:3_res_False_use_jk:Truescore:0.8304.ckpt.res",
}

if __name__ == '__main__':
    for dataset in info:
        save_path = info[dataset][:-4]
        # disturb_info = {"AEHGN": info[dataset]}
        disturb_info = {"base": [info[dataset]]}
        for type in ["origin", "all", "age", "gender"]:
            disturb_info[type] = []
            for run in range(10):
                disturb_info[type].append(save_path + type + str(run) + ".res")

        save_file = f"{dataset}_disturb_metric_path_v4.pth"
        force = False

        if os.path.exists(save_file) and not force:
            metric_dict = pk.load(open(save_file, "rb"))
        else:
            metric_dict = {}
            for model in disturb_info:
                metric_dict[model] = {}
                for i in range(len(disturb_info[model])):
                    filename = disturb_info[model][i]
                    print("loading:", model)
                    res = torch.load(filename)
                    res = merge_res(res)

                    top = 10
                    metrics = cal_auc_sperately(res[0], res[1], top_k=top)

                    print("AUC:", metrics[0])
                    print("Precision:", metrics[1])
                    print("Recall:", metrics[2])

                    metric_dict[model][i] = metrics

            pk.dump(metric_dict, open(save_file, "wb"))

    for dataset in info:
        save_file = f"{dataset}_disturb_metric_path_v4.pth"
        metric_dict = pk.load(open(save_file, "rb"))
        dataset_source = PLEASESource()
        se_map = dataset_source.se_map
        se_map_rev = dataset_source.se_map_rev
        metric_name = ["AUC", "Precision", "Recall@10"]

        for metric_index in [0, 2]:
            csv_data = []
            n_se = len(metric_dict["origin"][0][0])
            for idx in range(n_se):
                SE_info = {}
                for type in ["base", "origin", "gender", "age", "all"]:
                    SE_info["SE_id"] = se_map_rev[idx]
                    SE_info["SE_name"] = se_name_map[SE_info["SE_id"]]
                    for run in range(len(metric_dict[type])):
                        SE_info[f"{type}-{run}"] = metric_dict[type][run][metric_index][idx]
                    SE_info[type] = np.mean([SE_info[f"{type}-{run}"] for run in range(len(metric_dict[type]))])
                SE_info["diff"] = SE_info["origin"] - max(SE_info["gender"], SE_info["age"], SE_info["all"])
                csv_data.append(SE_info)

            data_pd = pd.DataFrame(csv_data)
            data_pd.sort_values(by="origin-0", ascending=False)
            print(dataset, metric_name[metric_index], data_pd[data_pd["diff"] >= -0.02])
            data_pd.to_csv(f"{dataset}_{metric_name[metric_index]}.csv")
            data_pd[data_pd["diff"] >= -0.02].to_csv(f"{dataset}_{metric_name[metric_index]}_ge_0.csv")
