from a2_filtering import pk, faers_root, scan_se_i_d, filter_reasonable2
import os

def filter_ses(data_pd, drug_ses):
    """

    :param data_pd:
    :param drug_ses:
    :return:
    """
    filtered_data = data_pd
    # 5. Exclude low-frequency SE (Side Effects), drugs, indications & meaningless SE.
    filtered_data["SE"] = filtered_data.apply(lambda x: [each for each in x["SE"] if each in drug_ses], axis=1)

    filtered_data = filtered_data[filtered_data.apply(lambda x: len(x["SE"]) > 0, axis=1)]

    return filtered_data


def build_v3(all_pd_US, drug_ses, save_path = f"{faers_root}/filtered_data_v3.pk"):

    if os.path.exists(save_path):
        filtered_data_US = pk.load(open(save_path, 'rb'))
    else:
        filtered_data_US = filter_ses(all_pd_US, drug_ses)
        # US data
        pk.dump(filtered_data_US, open(save_path, 'wb'))
        print("Input data info:", len(filtered_data_US))
        scan_se_i_d(filtered_data_US)
        print()

    # filtered_data_US = filter_reasonable2(filtered_data_US)
    # US data
    pk.dump(filtered_data_US, open(save_path, 'wb'))
    print(f"{save_path} info:", len(filtered_data_US))
    scan_se_i_d(filtered_data_US)
    print()


if __name__ == '__main__':
    all_pd_US = pk.load(open(f"{faers_root}/filtered_data_v1.pk", 'rb'))
    drug_ses = pk.load(open("./drugs_SE_set.pk", "rb"))
    drug_ses = {str(each) for each in drug_ses}
    build_v3(all_pd_US, drug_ses)

    all_pd = pk.load(open(f"{faers_root}/filtered_data_v2.pk", 'rb'))
    build_v3(all_pd, drug_ses, save_path=f"{faers_root}/filtered_data_v4.pk")