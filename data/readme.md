## 0. Setup
Download and pre-proces data from FAERS.
"AE_dic.pk",  "AE_mapping.pk", and "drug_mapping.pk" need manually downloaded from "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/G9SHDA".
```shell
bash download_all.sh
python a1_preprocessing.py
```
The output of 'a1_preprocessing.py' is 'patient_safety_2022.pk', which consists of the AE reports from 2013 to 2022.

## 1. Quality Controlling

1. Filter American data.
1. Filter indications, drugs, and side effects with frequencies greater than 100.
1. Select reports with Quality ratings of [1, 2, 3], while keeping those where the length of side effects (AE), indications, and drugs is greater than 0.

```shell
python a2_filtering.py
```
The generated records are store in "data/Data/curated/patient_safety_2022.pk".

4. Furthermore, we have filtered out adverse events (AEs), indications, and drugs that appear more than 100 times. 
The results are saved in the file named "filtered_data_v1.pk."
Note "filtered_data_v2.pk" contains AE reports meets requires 2~4. In other words, "filtered_data_v2.pk" consists AE reports from American and other countries.

Codes are modified according to https://github.com/mims-harvard/patient-safety.

## 2. Drug Interference
Identify adverse events (AEs) related to Drugs \[and Indication\], with a preliminary scope.

```shell
python [b1_build_data.py](b1_build_data.py)
python [b2_cal_CI_se_drug.py](b2_cal_CI_se_drug.py)
python [b3_analyse_significant_drugs.py](b3_analyse_significant_drugs.py)
python [b4_analyse_csv2_drug.py](b4_analyse_csv2_drug.py)
```
Filter AE reports with drug related AEs.
```shell
python [b5_filter_by_drug_ses.py](b5_filter_by_drug_ses.py)
```
The results are store in "filtered_data_v3.pk."

## 3. Association Mining in [association_mining](..%2Fassociation_mining)

```shell
python [c1_filter_by_drug_ses.py](c1_filter_by_drug_ses.py)
```

