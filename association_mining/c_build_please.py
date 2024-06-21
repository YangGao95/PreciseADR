# %%
# 1. load data

from data.a2_filtering import *

base_path = osp.dirname(__file__)
faers_root = f"{base_path}/../data/Data/curated"


# faers_root = f"{base_path}/../association_mining"


def get_age_related_se():
    return pk.load(open(f"./age_SE_set.pk", "rb"))


def get_gender_related_se():
    return pk.load(open(f"./gender_SE_set.pk", "rb"))


def filter_by_ses(data, ses, savefile):
    data["SE"] = data.apply(lambda x: [each for each in x["SE"] if each in ses], axis=1)
    data = data[data.apply(lambda x: len(x["SE"]) > 0, axis=1)]

    data = data.sort_values(by='receipt_date')
    pickle.dump(data, open(savefile, 'wb'))
    print(f"{savefile} info:", len(data))
    scan_se_i_d(data)
    print()


if __name__ == '__main__':
    # age
    all_pd_US = pickle.load(open(f"{faers_root}/filtered_data_v3.pk", 'rb'))
    all_pd = pickle.load(open(f"{faers_root}/filtered_data_v4.pk", 'rb'))

    SE_set = get_age_related_se()
    SE_set = {str(each) for each in SE_set}
    print(len(SE_set))

    filter_by_ses(all_pd_US, SE_set, f"{faers_root}/PLEASE-US-age.pk")
    filter_by_ses(all_pd, SE_set, f"{faers_root}/PLEASE-ALL-age.pk")

    # gender
    SE_set = get_gender_related_se()
    SE_set = {str(each) for each in SE_set}
    print(SE_set)
    print(len(SE_set))

    filter_by_ses(all_pd_US, SE_set, f"{faers_root}/PLEASE-US-gender.pk")
    filter_by_ses(all_pd, SE_set, f"{faers_root}/PLEASE-ALL-gender.pk")
