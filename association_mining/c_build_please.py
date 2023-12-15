# %%
# 1. load data

from data.a2_filtering import *


base_path = osp.dirname(__file__)
faers_root = f"{base_path}/../data/Data/curated"


def get_age_related_se():
    return pk.load(open(f"./age_SE_set.pk", "rb"))


def get_gender_related_se():
    return pk.load(open(f"./gender_SE_set.pk", "rb"))



if __name__ == '__main__':
    # age
    all_pd_US = pickle.load(open(f"{faers_root}/filtered_data_v3.pk", 'rb'))
    SE_set = get_age_related_se()
    SE_set = {str(each) for each in SE_set}
    print(len(SE_set))

    all_pd_US["SE"] = all_pd_US.apply(lambda x: [each for each in x["SE"] if each in SE_set], axis=1)
    all_pd_US = all_pd_US[all_pd_US.apply(lambda x: len(x["SE"]) > 0, axis=1)]

    all_pd_US = all_pd_US.sort_values(by='receipt_date')
    pickle.dump(all_pd_US, open(f"{faers_root}/PLEASE-US-age.pk", 'wb'))
    print("PLEASE-US-age info:", len(all_pd_US))
    scan_se_i_d(all_pd_US)
    print()

    # gender
    all_pd_US = pickle.load(open(f"{faers_root}/filtered_data_v3.pk", 'rb'))
    SE_set = get_gender_related_se()
    SE_set = {str(each) for each in SE_set}
    print(SE_set)

    all_pd_US["SE"] = all_pd_US.apply(lambda x: [each for each in x["SE"] if each in SE_set], axis=1)
    all_pd_US = all_pd_US[all_pd_US.apply(lambda x: len(x["SE"]) > 0, axis=1)]

    all_pd_US = all_pd_US.sort_values(by='receipt_date')
    pickle.dump(all_pd_US, open(f"{faers_root}/PLEASE-US-gender.pk", 'wb'))
    print("PLEASE-US-gender info:", len(all_pd_US))
    scan_se_i_d(all_pd_US)
    print()

    # aim
    all_pd_US = pickle.load(open(f"{faers_root}/filtered_data_v3.pk", 'rb'))
    SE_set = get_gender_related_se()
    SE_set_2 = get_age_related_se()
    SE_set = SE_set.union(SE_set_2)
    SE_set = {str(each) for each in SE_set}
    print(SE_set)

    all_pd_US["SE"] = all_pd_US.apply(lambda x: [each for each in x["SE"] if each in SE_set], axis=1)
    all_pd_US = all_pd_US[all_pd_US.apply(lambda x: len(x["SE"]) > 0, axis=1)]

    all_pd_US = all_pd_US.sort_values(by='receipt_date')
    pickle.dump(all_pd_US, open(f"{faers_root}/PLEASE-US-aim.pk", 'wb'))
    print("PLEASE-US-aim info:", len(all_pd_US))
    scan_se_i_d(all_pd_US)
    print()

