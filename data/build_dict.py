import pickle
import os

base_path = os.path.dirname(__file__)


def build_dict():
    global se_map
    se_dic = pickle.load(open(f'{base_path}/./Data/curated/AE_dic.pk', 'rb'))
    se_map = {v[0]: k for k, v in se_dic.items()}

    drug_dic = pickle.load(open(f'{base_path}/./Data/curated/drug_mapping.pk', 'rb'))
    drug_map = {v[0]: k for k, v in drug_dic.items()}
    with open(f'{base_path}/./Data/drug_map.pk', "wb") as f:
        pickle.dump(drug_map, f)
    with open(f'{base_path}/./Data/se_map.pk', "wb") as f:
        pickle.dump(se_map, f)

    return drug_map, se_map


if not os.path.exists(f'{base_path}/./Data/drug_map.pk') or not os.path.exists(f'{base_path}/./Data/se_map.pk'):
    build_dict()

with open(f'{base_path}/Data/drug_map.pk', "rb") as f:
    drug_map = pickle.load(f)

with open(f'{base_path}/Data/se_map.pk', "rb") as f:
    se_map = pickle.load(f)


