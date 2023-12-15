# a2_filtering.py : Quality Controlling,

# 1. load data
import pickle as pk
import pickle
import os.path as osp
import numpy as np
import pandas as pd
from functools import partial
from data.meanless_se import drop_list

base_path = osp.dirname(__file__)
faers_root = f"{base_path}/Data/curated"


def scan_se_i_d(data_pd):
    """
    Statistics of information related to SE (Side Effects), Indication, and Drug.
    :param data_pd:
    :return:
    """
    SE_list, indication_list, drug_list = [], [], []

    def scan_attr(x, key="SE", res_list=SE_list):
        res_list.extend(x[key])

    scan_se = partial(scan_attr, key="SE", res_list=SE_list)
    scan_drug = partial(scan_attr, key="drugs", res_list=drug_list)
    scan_indication = partial(scan_attr, key="indications", res_list=indication_list)

    data_pd.apply(scan_se, axis=1)
    data_pd.apply(scan_drug, axis=1)
    data_pd.apply(scan_indication, axis=1)

    SE_set, indication_set, drug_set = set(SE_list), set(indication_list), set(drug_list)

    print("SE size:", len(SE_set))
    print("drug size:", len(drug_set))
    print("Indication size:", len(indication_set))

    return SE_list, indication_list, drug_list, SE_set, indication_set, drug_set


def basic_filter(x):
    """
    Filtering data that meets the requirements.
    :param x:
    :return:
    """
    gender = float(x["gender"])
    age = float(x["age"])
    weight = float(x["weight"])
    return gender > 0 and 120 >= age > 0 and 400 >= weight > 0


def get_selected_SE_d_i(SE_list, drug_list, indication_list, threshold=[100, 100, 100]):
    """
    filter out SE (Side Effects), Drug, and Indication occurrences that exceed the threshold frequency,
    and return the sets of SE, Drug, and Indication.
    :param SE_list:
    :param drug_list:
    :param indication_list:
    :param threshold:
    :return:
    """
    SE_t, d_t, i_t = threshold
    se_name, se_count = np.unique(SE_list, return_counts=True)
    drug_name, drug_count = np.unique(drug_list, return_counts=True)
    indication_name, indication_count = np.unique(indication_list, return_counts=True)
    SE_list, SE_count = [], []

    for name, count in zip(se_name, se_count):
        if name not in drop_list and count > SE_t:
            SE_list.append(name)
            SE_count.append(count)

    print("Remain SE len", len(SE_list))

    drug_set = set()
    for name, count in zip(drug_name, drug_count):
        if count > d_t:
            drug_set.add(name)

    indication_set = set()
    for name, count in zip(indication_name, indication_count):
        if count > i_t:
            indication_set.add(name)

    SE_set = set(SE_list)

    return SE_set, drug_set, indication_set


def filter_reasonable2(data_pd, threshold=[100, 100, 100]):
    """
    Set a threshold, retain the given data, keep non-empty data; discard the AE (Adverse Event), Indication,
    or Drug if it's not in the set.
    :param data_pd:
    :param threshold:
    :return:
    """
    SE_list, indication_list, drug_list, SE_set, indication_set, drug_set = scan_se_i_d(data_pd)
    SE_set, drug_set, indication_set = get_selected_SE_d_i(SE_list, drug_list, indication_list, threshold)
    filtered_data = data_pd
    # 5. Exclude low-frequency SE (Side Effects), drugs, indications & meaningless SE
    filtered_data["SE"] = filtered_data.apply(lambda x: [each for each in x["SE"] if each in SE_set], axis=1)
    drug_mask = filtered_data.apply(lambda x: np.array([each in drug_set for each in x["drugs"]]).all(), axis=1)
    filtered_data = filtered_data[drug_mask]
    i_mask = filtered_data.apply(lambda x: np.array([each in indication_set for each in x["indications"]]).all(),
                                 axis=1)
    filtered_data = filtered_data[i_mask]

    # keep data that #indication, #drug, #SE >1
    filtered_data = filtered_data[filtered_data.apply(lambda x: len(x["SE"]) > 0, axis=1)]
    # filter indications
    filtered_data = filtered_data[filtered_data.apply(lambda x: len(x["indications"]) > 0, axis=1)]
    # filter drugs
    filtered_data = filtered_data[filtered_data.apply(lambda x: len(x["drugs"]) > 0, axis=1)]

    return filtered_data


def build_v2():
    filtered_data_US = filter_reasonable2(all_pd_US_pro)
    # US data
    pickle.dump(filtered_data_US, open(f"{faers_root}/filtered_data_v1.pk", 'wb'))
    print("filtered_data_v3 info:")
    scan_se_i_d(filtered_data_US)
    print()

    # All data
    filtered_data = filter_reasonable2(all_data_pd)
    pickle.dump(filtered_data, open(f"{faers_root}/filtered_data_v2.pk", 'wb'))
    print("filtered_data_v4 info:")
    scan_se_i_d(filtered_data)
    print()


if __name__ == '__main__':
    # %%
    # 2. Statistics of input data
    all_data_pd = pk.load(open(f"{faers_root}/patient_safety_2022.pk", "rb"))
    print("origin data size:", all_data_pd.shape)
    SE_list, indication_list, drug_list, SE_set, indication_set, drug_set = scan_se_i_d(all_data_pd)

    # %%
    ## 3. US data, qualify in ['1', '2', '3']

    all_pd_US = all_data_pd[all_data_pd.country == 'US'].copy()
    print('Focus on US, reports #', all_pd_US.shape)

    all_pd_US_pro = all_pd_US[all_pd_US.qualify.isin(['1', '2', '3'])]  # professional: 1,2,3
    all_pd_US_pro = all_pd_US_pro[all_pd_US_pro.apply(basic_filter, axis=1)]
    print('Focus on professional qualification, reports #', all_pd_US_pro.shape)
    pickle.dump(all_pd_US_pro, open(f"{faers_root}/all_pd_US_pro.pk", 'wb'))

    print("US_pro data:")
    SE_list_US, indication_list_US, drug_list_US, SE_set_US, indication_set_US, drug_set_US = scan_se_i_d(all_data_pd)
    print()

    # %%
    ## 3-2 all_pd, qualify in ['1', '2', '3']

    all_pd_pro = all_data_pd[all_data_pd.qualify.isin(['1', '2', '3'])]  # professional: 1,2,3
    all_pd_pro = all_pd_pro[all_pd_pro.apply(basic_filter, axis=1)]
    print('Focus on professional qualification, reports #', all_pd_pro.shape)
    pickle.dump(all_pd_pro, open(f"{faers_root}/all_pd_pro.pk", 'wb'))

    print("All data data info:")
    SE_list, indication_list, drug_list, SE_set, indication_set, drug_set = scan_se_i_d(all_data_pd)
    print()

    # %%
    # 4. build dataset
    build_v2()
