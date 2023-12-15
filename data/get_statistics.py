from data.a2_filtering import *

if __name__ == '__main__':

    # 1. downloaded
    all_pd_US = pickle.load(open(f"{faers_root}/patient_safety_2022.pk", 'rb'))
    print("all_pd_US_pro info:")
    print(all_pd_US.shape)
    scan_se_i_d(all_pd_US)

    # 2. After Quality Controlling

    filtered_data_US = pickle.load(open(f"{faers_root}/filtered_data_v1.pk", 'rb'))
    print("filtered_data_v1 info:")
    print(filtered_data_US.shape)
    scan_se_i_d(filtered_data_US)

    # 3. After Drug Interference
    filtered_data_US = pickle.load(open(f"{faers_root}/filtered_data_v3.pk", 'rb'))
    print("filtered_data_v3 info:")
    print(filtered_data_US.shape)
    scan_se_i_d(filtered_data_US)

    # 4. Age-Related
    filtered_data_US = pickle.load(open(f"{faers_root}/PLEASE-US-age.pk", 'rb'))
    print("filtered_data_v3 info:")
    print(filtered_data_US.shape)
    scan_se_i_d(filtered_data_US)

