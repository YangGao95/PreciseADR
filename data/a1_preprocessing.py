# a1_preprocessing.py: parse data and restore to pickle file.

# %load dic_reading.py
import glob
import os.path
import pickle
import xml.etree.ElementTree as ET
from datetime import date, datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

se_dic = pickle.load(open('./Data/curated/AE_dic.pk', 'rb'))
drug_dic = pickle.load(open('./Data/curated/drug_mapping.pk', 'rb'))

# In this MeDRA_dic, key is string of PT_name, value is a list:
# [PT, PT_name, HLT,HLT_name,HLGT,HLGT_name,SOC,SOC_name,SOC_abbr]
# meddra_pd_all = pickle.load(open('./Data/curated/AE_mapping.pk', 'rb'))
meddra_pd_all = se_dic


# %%
# initial setup
def date_normalize(formate, dat):
    stand_date = date(2000, 1, 1)
    if formate == '102':  # the date is formed as yyyymmdd
        current_date = date(int(dat[:4]), int(dat[4:6]), int(dat[6:8]))
    elif formate == '610':  # formed as yyyymm
        current_date = date(int(dat[:4]), int(dat[4:6]), 1)
    elif formate == '602':  # formed as yyyy
        current_date = date(int(dat[:4]), 1, 1)
    delta = current_date - stand_date
    return delta.days


def days_to_date(days):
    stand_date = date(2000, 1, 1)

    if int(days) < 0:
        days = 1

    dt = datetime.fromordinal(int(days))
    return dt.strftime('%Y-%m-%d')


# %% md
# Parse XML files
# %%
# initial setup
def date_normalize(formate, dat):
    stand_date = date(2000, 1, 1)
    if formate == '102':  # the date is formed as yyyymmdd
        current_date = date(int(dat[:4]), int(dat[4:6]), int(dat[6:8]))
    elif formate == '610':  # formed as yyyymm
        current_date = date(int(dat[:4]), int(dat[4:6]), 1)
    elif formate == '602':  # formed as yyyy
        current_date = date(int(dat[:4]), 1, 1)
    delta = current_date - stand_date
    return delta.days


n_reports = []
miss_count = {}

for yr in range(2013, 2023):

    qtr_list = [1, 2, 3, 4]
    for qtr in qtr_list:
        qtr_name = str(yr) + 'q' + str(qtr)
        print('I am parsing:', qtr_name)

        save_filename = './Data/parsed/' + qtr_name + '.pk'
        if os.path.exists(save_filename):
            continue

        #         """Read data from lab storage"""
        lab_storage = './'
        files = lab_storage + qtr_name + '/**/**'
        xml_files = glob.glob(files + "/*.xml", recursive=True)
        unique_files = list(set(xml_files))  # only keep the unique values, remove duplicated files.
        xml_files = unique_files
        xml_files.sort()
        print('find {} files'.format(len(xml_files)))
        print(xml_files)

        root = None
        for xml_file in xml_files:
            print(xml_file)
            data = ET.parse(xml_file).getroot()
            if root is None:
                root = data
            else:
                root.extend(data)
                print('finished merge', xml_file)
        nmb_reports = len(root)
        print(nmb_reports)

        count = 0
        patient_ID = 0
        dic = {}

        miss_admin = miss_patient = miss_reaction = miss_drug = 0
        for report in tqdm(root.findall('safetyreport')):
            """Administrative Information"""
            #             report.find('').text
            try:  # Mandatory Information: report_id
                try:
                    version = report.find('safetyreportversion').text
                except:
                    version = '1'

                report_id = report.find('safetyreportid').text

                try:
                    case_id = report.find('companynumb').text
                except:
                    case_id = '0'  # unknown case id

                try:
                    country = report.find('primarysource')[0].text
                except:
                    country = 'unknown'

                if country == 'COUNTRY NOT SPECIFIED':
                    country = 'unknown'

                try:
                    qualify = report.find('primarysource')[1].text
                except:
                    qualify = '6'  # the qualify is unknown

                #                 qualify = report.find('primarysource')[1].text

                if qualify not in {'1', '2', '3', '4', '5', '6', '7'}:
                    qualify = '0'

                try:
                    serious = report.find('serious').text
                except:
                    serious = '-1'

                try:
                    s_1 = report.find('seriousnessdeath').text
                except:
                    s_1 = '0'
                try:
                    s_2 = report.find('seriousnesslifethreatening').text
                except:
                    s_2 = '0'
                try:
                    s_3 = report.find('seriousnesshospitalization').text
                except:
                    s_3 = '0'
                try:
                    s_4 = report.find('seriousnessdisabling').text
                except:
                    s_4 = '0'
                try:
                    s_5 = report.find('seriousnesscongenitalanomali').text
                except:
                    s_5 = '0'
                try:
                    s_6 = report.find('seriousnessother').text
                except:
                    s_6 = '0'
                serious_subtype = [s_1, s_2, s_3, s_4, s_5, s_6]
            except:
                miss_admin += 1
                continue

            try:  # Optional information
                # receivedate: Date when the report was the FIRST received
                receivedateformat, receivedate = report.find('receivedateformat').text, report.find('receivedate').text
                receivedate = date_normalize(receivedateformat, receivedate)
            except:
                receivedate = '0'

            try:
                # receiptdate: Date of most RECENT report received
                receiptdateformat, receiptdate = report.find('receiptdateformat').text, report.find('receiptdate').text
                receiptdate = date_normalize(receiptdateformat, receiptdate)
            except:
                receiptdate = '0'

            for patient in report.findall('patient'):
                """Demographic Information"""
                try:
                    age = patient.find('patientonsetage').text
                except:
                    age = -1  # unknown age
                try:
                    ageunit = patient.find('patientonsetageunit').text
                except:
                    ageunit = '801'
                    # normalize age
                try:
                    age = int(age)
                    if age != -1:
                        if ageunit == '800':  # Decade
                            age = '-1'
                        elif ageunit == '801':  # Year
                            age = age
                        elif ageunit == '802':  # Month
                            age = int(age / 12)
                        elif ageunit == '803':  # Week
                            age = int(age / 52)
                        elif ageunit == '804':  # Day
                            age = int(age / 365)
                        elif ageunit == '805':  # Hour
                            age = int(age / (24 * 365))
                #                     else:
                #                         age = '-1'  # unknown age
                except:
                    age = -1

                try:
                    gender = patient.find('patientsex').text
                except:
                    gender = '0'
                try:
                    weight = patient.find('patientweight').text
                except:
                    weight = '0'
                ## Nothing is mandatory
                #                 if age == -1 and gender== '0':  # Mandatory: if age & gender both missing, ignore this report.
                #                     miss_patient +=1
                #                     continue

                reaction_list = []
                for side_ in patient.findall('reaction'):
                    try:  # outcome: 1-6, 6 levels in total
                        try:
                            PT_code = side_[0].text
                        except:
                            PT_code = '0'
                        try:
                            outcome = side_[2].text
                        except:
                            outcome = '6'
                        try:
                            PT = side_[1].text
                        except:
                            PT = 'none'
                        reaction = [PT_code, PT, outcome]
                    except:
                        continue
                    reaction_list.append(reaction)
                if reaction_list.__len__() == 0:  # Mandatory condition: at least has one reaction
                    miss_reaction += 1
                    continue

                drug_list = []
                for drug_ in patient.findall('drug'):
                    try:
                        try:
                            char = drug_.find(
                                'drugcharacterization').text  # drugcharacterization: 1(suspect)/2(concomitant)/3(interacting)
                        except:
                            char = '0'
                        try:
                            product = drug_.find('medicinalproduct').text  # drug brand
                        except:
                            product = 'none'
                        """Dosage are generally fixed according to the indication"""
                        try:
                            dorse, unit = drug_.find('drugstructuredosagenumb').text, drug_.find(
                                'drugstructuredosageunit').text
                            drugseparatedosagenumb, drugintervaldosageunitnumb, drugintervaldosagedefinition = \
                                drug_.find('drugseparatedosagenumb').text, drug_.find(
                                    'drugintervaldosageunitnumb').text, \
                                drug_.find('drugintervaldosagedefinition').text
                            form = drug_.find('drugdosageform').text  # tablet or capsule or sth
                        except:
                            dorse, unit, drugseparatedosagenumb, drugintervaldosageunitnumb, drugintervaldosagedefinition, form = \
                                '0', '0', '0', '0', '0', '0'
                        try:
                            route = drug_.find('drugadministrationroute').text
                            if route == '048':
                                route = '1'  # oral
                            elif route == '061':
                                route = '2'  # Topical
                        except:
                            route = '0'  # no information of route

                        try:
                            indication = drug_.find('drugindication').text  # indication (disease): super important
                        except:
                            indication = 'none'

                        try:
                            start_format, start_date = drug_.find('drugstartdateformat').text, drug_.find(
                                'drugstartdate').text
                            start_date = date_normalize(start_format, start_date)
                        except:
                            start_date = '0'
                        try:
                            end_format, end_date = drug_.find('drugenddateformat').text, drug_.find('drugenddate').text
                            end_date = date_normalize(end_format, end_date)
                        except:
                            try:
                                end_date = receiptdate
                            except:
                                end_date = '0'

                        try:
                            action = drug_.find('actiondrug').text
                        except:
                            action = '5'
                        try:
                            additional = drug_.find('drugadditional').text
                        except:
                            additional = '3'
                        try:
                            readm = drug_.find('drugrecurreadministration').text
                        except:
                            readm = '3'
                        try:
                            substance = drug_.find('activesubstance')[0].text
                        except:
                            substance = 'none'
                    except:  # Mandatory condition: if none of the above information is provided, ignore this report
                        continue
                    drug = [char, product, dorse, unit, drugseparatedosagenumb, drugintervaldosageunitnumb,
                            drugintervaldosagedefinition, form, route, indication, start_date, end_date, action,
                            readm, additional, substance]
                    drug_list.append(drug)
                if drug_list.__len__() == 0:
                    miss_drug += 1
                    continue

                """for patient_ID"""
                dic[count] = [version, report_id, case_id, country, qualify, serious,
                              s_1, s_2, s_3, s_4, s_5, s_6,
                              receivedate, receiptdate,
                              age, gender, weight, reaction_list, drug_list]
                count += 1

        pickle.dump(dic, open('./Data/parsed/' + qtr_name + '.pk', 'wb'))

        n_reports.append(len(dic))
        print(qtr_name + ' file saved. with', len(dic), 'reports')
        miss_count[qtr_name] = [nmb_reports, miss_admin, miss_patient, miss_reaction, miss_drug]

print('All data saved')

# %% md
# Merge files
# %%
ho_combos = {}

nmb = 0
n_reports = []
for yr in range(2013, 2023):
# for yr in range(2019, 2020):

    qtr_list = [1, 2, 3, 4]
    for qtr in qtr_list:
        qtr_name = str(yr) + 'q' + str(qtr)
        file_path = './Data/parsed/' + qtr_name + '.pk'
        print('I am loading {} from {}'.format(qtr_name, file_path))
        dic = pickle.load(open(file_path, 'rb'))
        n_reports.append(len(dic))
        print('loaded', len(dic))
        values = np.array(list(dic.values()), dtype=object)
        print(values.shape)

        admin_demo = values[:, :17]
        # Find all reactions/side effects in this quarter
        se = values[:, 17]
        drugs = values[:, 18]
        for i in range(values.shape[0]):  # dive into a single report
            new_se = []
            new_drug = []  # empty mapped se and drugs
            new_indication = []

            se_report = se[i]
            drugs_report = drugs[i]
            for j in range(len(se_report)):  # dive into a single reaction
                se_reaction = se_report[j][1].lower()
                if '\\' in se_reaction:
                    se_reaction = se_reaction.split('\\')[0]
                if se_reaction in se_dic:
                    se_key = se_reaction
                elif ' ' in se_reaction:
                    se_key_sets = se_reaction.split(' ')
                    if se_key_sets[0] in se_dic:
                        se_key = se_key_sets[0]
                    elif se_key_sets[1] in se_dic:
                        se_key = se_key_sets[1]
                try:
                    new_se.append(meddra_pd_all[se_reaction][0])  # Use MedDRA ID.
                except:  # if the key not in se_dic, continue
                    continue

            for k in range(len(drugs_report)):  # dive into a single drug
                drug = drugs_report[k][-1].lower()  # the drug substance
                indication = drugs_report[k][9].lower()  # the indication names

                # find the proper drug_key that exist in drug_dic
                if '\\' in drug:
                    drug = drug.split('\\')[0]
                if drug in drug_dic:
                    drug_key = drug
                elif ' ' in drug:
                    key_sets = drug.split(' ')
                    if key_sets[0] in drug_dic:
                        drug_key = key_sets[0]

                    elif key_sets[1] in drug_dic:
                        drug_key = key_sets[1]
                try:
                    #                     new_drug.append(drug_dic[drug_key][1])   # use code
                    new_drug.append(drug_dic[drug_key][0])  # use drugbank ID
                except:  # if the key not in drug_dic, continue
                    continue

                # find the proper drug_key that exist in drug_dic
                # Using meddra_pd_all
                if '\\' in indication:
                    indication = indication.split('\\')[0]
                if indication in meddra_pd_all:
                    indication_key = indication
                elif ' ' in indication:
                    key_sets = indication.split(' ')
                    if key_sets[0] in meddra_pd_all:
                        indication_key = key_sets[0]

                    elif key_sets[1] in meddra_pd_all:
                        indication_key = key_sets[1]
                try:
                    new_indication.append(meddra_pd_all[indication_key][0])  # use MedDRA ID
                except:  # if the key not in drug_dic, continue
                    continue

            # Feed into dictionary
            new_se.sort()
            new_drug.sort()  # keep the new_se and new_drug sorted, incase [1,2] and [2,1] appears as different drug list
            new_indication.sort()

            xx = list(admin_demo[i])
            xx.append(new_se)
            xx.append(new_drug)
            xx.append(new_indication)
            ho_combos[nmb] = xx  # [admin_demo[i], new_se, new_drug, new_indication]
            nmb += 1

print('#- all reports', sum(n_reports))
print('The No. of high order combos:', len(ho_combos))
pickle.dump(ho_combos, open('./Data/parsed/reports_v4.pk', 'wb'))
# Please revise 'Data/processed_2/v4/' to 'Data/parsed/'

print('ho_combos saved', ho_combos.get(0))
# %% md
# Make DataFrame

## Remove duplicate case_id and report_id

# %%
reports_v4 = pickle.load(open('./Data/parsed/reports_v4.pk', 'rb'))
len(reports_v4)
# %%
reports_pd = pd.DataFrame(reports_v4.values(),
                          columns=['version', 'report_id', 'case_id', 'country', 'qualify', 'serious',
                                   's1', 's2', 's3', 's4', 's5', 's6', 'receivedate', 'receiptdate',
                                   'age', 'gender', 'weight', 'SE', 'drugs', 'indications'])

# reports_pd['lastingdays'] = reports_pd['receiptdate'] - reports_pd['receivedate']
# %%
# check the #- of headache
ix = ['10019211' in j for j in reports_pd.SE]
print('#-of headache', sum(ix))
"""# of drugs and AEs"""
import itertools

n_se = len(set(list(itertools.chain(*reports_pd.SE))))
n_drug = len(set(list(itertools.chain(*reports_pd.drugs))))
n_indication = len(set(list(itertools.chain(*reports_pd.indications))))
print(f'#-SE: {n_se}, #-drugs: {n_drug}, #-indication: {n_indication} , in #-reports:{len(reports_pd)}')
# %%
# this command is time-consumming.
reports_pd['date'] = reports_pd.apply(lambda row: str(pd.Period('2000-01-01') + int(row['receivedate'])), axis=1)
reports_pd.head()
# %%
print('#-of all reports', reports_pd.shape)
reports_pd = reports_pd.drop_duplicates(['report_id', 'case_id', 'receivedate'], keep="last")
print('After remove duplicate reports', reports_pd.shape)

# reports_pd = reports_pd.drop_duplicates(['case_id'], keep="last")
# print('After remove duplicate case_id',reports_pd.shape)
# %%
# check the #- of headache
ix = ['10019211' in j for j in reports_pd.SE]
print('#-of headache', sum(ix))
# %%
pickle.dump(reports_pd, open('./Data/parsed/reports_v4_pd_new.pk', 'wb'))
print('reports_v4_pd_new saved', reports_pd.shape)
# %% md
##  Add receiptdate
# %%
new_pd = pickle.load(open('./Data/parsed/reports_v4_pd_new.pk', 'rb'))
print('reports_v4_pd_new saved', new_pd.shape)
# %%
new_pd['receipt_date'] = new_pd.apply(lambda row: str(pd.Period('2000-01-01') + int(row['receiptdate'])), axis=1)
new_pd.head()
# %%
pickle.dump(new_pd, open('./Data/curated/patient_safety_2022.pk', 'wb'))

print('patient_safety saved', new_pd.shape)
