# Association Mining
## Input files:


## Process:

[c1_build_data_pair.py](c1_build_data_pair.py) Counts occurrences of Drug-SE pairs.
[c2_cal_CI.py](c2_cal_CI.py) Calculates the confidence interval for Drug-SE ROR (Reporting Odds Ratio).
[c3_analyse_significant_3_v2.py](c3_analyse_significant_3_v2.py) Identifies significant differences among Drug-SE pairs based on ROR and confidence intervals and records them in a file.
[c4_analyse_csv2.py](c4_analyse_csv2.py) Analyse the SE_set and Drug-SE pairs.
[c5_merge_se.py](c5_merge_se.py) Merges age SE and gender SE.
[c_build_please.py](c_build_please.py) Build the PLEASE dataset.
[to_csv.py](to_csv.py) Write significant Drug-SE pairs along with their associated ROR values to a csv file.


## Output files:
age_SE_set.pk : Age Related ADRs
gender_SE_set.pk : Gender Related ADRs
PLEASE-US-age.pk # PLEASE-age dataset for US
PLEASE-US-gender.pk # PLEASE-gender dataset for US