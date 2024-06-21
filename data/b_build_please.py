# %%
# 1. load data

from data.a2_filtering import *


base_path = osp.dirname(__file__)
faers_root = f"{base_path}/Data/curated"


def get_age_related_se():
    return pk.load(open(f"{base_path}/../association_mining/age_SE_set.pk", "rb"))


def get_gender_related_se():
    return pk.load(open(f"{base_path}/../association_mining/gender_SE_set.pk", "rb"))



if __name__ == '__main__':
    # all
    all_pd_US = pickle.load(open(f"{faers_root}/filtered_data_v3.pk", 'rb'))
    pickle.dump(all_pd_US, open(f"{faers_root}/PLEASE-US-all.pk", 'wb'))
    print("PLEASE-US-all info:", len(all_pd_US))
    scan_se_i_d(all_pd_US)
    print()

    all_pd = pickle.load(open(f"{faers_root}/filtered_data_v4.pk", 'rb'))
    pickle.dump(all_pd, open(f"{faers_root}/PLEASE-ALL-all.pk", 'wb'))
    print("PLEASE-ALL-all info:", len(all_pd))
    scan_se_i_d(all_pd)
    print()
