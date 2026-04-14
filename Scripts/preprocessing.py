#!/usr/local/anaconda3-2024.10-1/bin/python3
import pandas as pd
import numpy as np

def if_not_future_date(df, columns):
    for column in columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
        future_dates = df[df[column] > '01FEB2026']
        if not future_dates.empty:
            print(f"Warning: Future dates found in column '{column}', in {df.name}")
            print(future_dates)
            unexpected = df.loc[df[column] > '01FEB2026', column].unique()
            print("Unexpected values:")
            print(unexpected)
#save filtered out patients dataset
            future_dates.to_csv(f'../../data/filtered_out/{df.name}/{df.name}_{column}.csv', index=False)
#change to nan
            df.loc[df[column] > '01FEB2026', column] = "NA"

def if_unexpected_values(df, column, expected_values):
    mask = df[column].isin(expected_values) | df[column].isna()
    unexpected_values = df[~mask]
    if not unexpected_values.empty:
        print(f"Warning: Unexpected values found in column '{column}', in {df.name}")
        print(unexpected_values)
        unexpected = df.loc[~mask, column].unique()
        print("Unexpected values:")
        print(unexpected)
#save filtered out patients dataset
        unexpected_values.to_csv(f'../../data/filtered_out/{df.name}/{df.name}_{column}.csv', index=False)
#change to nan
        df.loc[~mask, column] = "NA"

#MASKSSSSSS

#masks excluding
#for only RA as diagnosis
RA_diagnosis = ['DIAGNOSIS_M05_9', 'DIAGNOSIS_M06_0', 'DIAGNOSIS_M06_9'] # df=everywhere but hospitals  #Diagnosis

review_state = ['private', 'inactive', 'NA'] # df=everywhere but hospitals  #Review_state_patient
#the oldest danish person 112
age = [str(i) for i in range(113)] + ["NA"] # df=everywhere but hospitals  #Age_at_diagnosis
#other masks
department_change = ["0", "1", "NA"]
sae_numbers = ["1'", "2'", "3'", "4'", "1", "2", "3", "4", "5_2011", "6_2011", "7_2011", "8_2016", "NA"]
sae_cont = ["NO", "YES", "NA"]
sae_relation = ["1'", "2'", "3'", "4'", "5'", "6'", "7'", "NA", "1", "2", "3", "4", "5", "6", "7"]
visit_type = ["ACUTE_EXTRA_MAR", "ACUTE_EXTRA_NUR", "ACUTE_EXTRA_PHY", "NA","ACUTE_EXTRA_MARKED", "ACUTE_EXTRA_NURSE", "ACUTE_EXTRA_PHYSIOTHERAPY"]
anchor = ["Better", "LittleBetter", "LittleWorse", "MuchWorse", "MuchBetter", "Unchanged", "Worse", "NA"]
Basdai = [f"{i/100:.2f}" for i in range(10001)] + ["NA"]
Basfi = [f"{i/100:.2f}" for i in range(10001)] + ["NA"]
Basmi = [f"{i/100:.2f}" for i in range(10001)] + ["NA"]
Haq = [f"{i/1000:.3f}" for i in range(3001)] + ["NA"]
VAS = [f"{i/100:.2f}" for i in range(10001)] + ["NA"]
questionaire = ["YES", "NO", "NA", "DON'T KNOW", "DONTKNOW"]
neg_pos = ["Neg", "Pos", "NA"]
koen = ["0", "1", "NA"]
diagnosis_type = ["Leddegigt", "Psoriasisgigt", "Rygsoejlegigt", "NA"]
intervention_type = ["Incident", "Praevalent", "Ukendt", "NA"]
RA_alert = ["INCREASED", "NEVER_POSSIBLE", "NOT_POSSIBLE", "OTHER", "PATIENT_SAY_NO", "nan", "NA"]

patients_multiple_diagnosis = set()
patients_wrong_diagnosis = set()

datasets = ["logistics", "patients", "treatments", "visits", "yearly_visits", "saes"]
filtered_datasets = {}
for dataset in datasets:
    df = pd.read_csv(f'../raw/{dataset}.csv', encoding = 'unicode_escape', dtype= str)
    X = df.copy()
    X = X.replace(",", ".", regex = True)
    X.name  = dataset
    print(f"Filtered out entries from dataset: {dataset}")

    if dataset in ["patients", "treatments", "visits", "yearly_visits", "saes", "logistics"]:
        if_not_future_date(X, ["Diagnosis_date"])
        if_unexpected_values(X, "afdeling_skift", department_change)
        if_unexpected_values(X, "Review_state_patient", review_state)
        if_unexpected_values(X, "Age_at_diagnosis", age)

    if dataset == "treatments": #prescriptions dataset
        if_not_future_date(X, ["Prescription_start_date", "Prescription_stop_date"])

    if dataset == "visits": #visits dataset
        if_not_future_date(X, ["Visit_date"])
        #masks for test values
        if_unexpected_values(X, "Visit_type", visit_type)
        if_unexpected_values(X, "Anchor", anchor)
        if_unexpected_values(X, "Basdai", Basdai)
        if_unexpected_values(X, "Basfi", Basfi)
        if_unexpected_values(X, "Basmi", Basmi)
        if_unexpected_values(X, "Haq", Haq)
        if_unexpected_values(X, "Vas_patient_global", VAS)
        if_unexpected_values(X, "Vas_patient_pain", VAS)
        if_unexpected_values(X, "Vas_patient_fatigue", VAS)
        if_unexpected_values(X, "Vas_doctor", VAS)
        if_unexpected_values(X, "Uveitis_since_last_visit", questionaire)
        if_unexpected_values(X, "Psoriasis_nail", questionaire)
        if_unexpected_values(X, "Psoriasis_dactylitis", questionaire)
        if_unexpected_values(X, "Psoriasis_dactylitis_020", [f"{i/100:.2f}" for i in range(2001)] + ["NA"])
        if_unexpected_values(X, "Swollenjoints28", [f"{i}" for i in range(29)] + ["NA"])
        if_unexpected_values(X, "Swollenjoints66", [f"{i}" for i in range(67)] + ["NA"])

    if dataset == "yearly_visits": # yearly visits dataset
        if_not_future_date(X, ["Xray_date", "Xray_columna_date", "MR_columna_date", "Xray_sacro_joint_date", "DXA_scanning_date", "MR_sacro_joint_date"])
        #masks for test values
        if_unexpected_values(X, "Year_of_status", [f"{i}" for i in range(1980, 2027)] + ["NA"])
        if_unexpected_values(X, "Anti_CCP", neg_pos)
        if_unexpected_values(X, "IgM_RF", neg_pos)

    if dataset == "saes": #saes dataset
        if_not_future_date(X, ["Sae_start_date", "Sae_stop_date"])
        if_unexpected_values(X, "Sae_number", sae_numbers)
        if_unexpected_values(X, "Sae_cont", sae_cont)
        if_unexpected_values(X, "Sae_relation", sae_relation)

    if dataset == "logistics": #logistics dataset
        if_not_future_date(X, ["Diagnosis_date_con", "Min_year_of_status", "Max_year_of_status", "diag_year", "CPR_status_dato", "Kontroldato_RA", "Kontroldato_SpA", "Basmi_date", "Xray_columna_date", "MR_colum_columna_date", "Xray_sacro_joint_date", "MR_sacro_joint_date", "DXA_scanning_date", "CRP_date", "Min_visit_date", "Max_visit_date", "Min_prescription_start_date", "Max_prescription_start_date", "Min_sae_start_date", "Max_sae_start_date", "Hop_dato_start","Vas_patient_pain_seneste_date", "Basdai_seneste_date", "Kontroldato_PsA", "Dato_DAPSA"])        #masks for test values
        if_unexpected_values(X, "Koen", koen)
        if_unexpected_values(X, "Age_at_diag_year", age)
        if_unexpected_values(X, "Diagnosis_type", diagnosis_type)
        if_unexpected_values(X, "Interventions_type", intervention_type)
        if_unexpected_values(X, "Haq", Haq)
        if_unexpected_values(X, "Vas_patient_pain", VAS)
        if_unexpected_values(X, "Basdai", Basdai)
        if_unexpected_values(X, "Basfi", Basfi)
        if_unexpected_values(X, "Basmi", Basmi)
        if_unexpected_values(X, "Anti_CCP", neg_pos)
        if_unexpected_values(X, "IgM_RF", neg_pos)
        if_unexpected_values(X, "Swollenjoints28", [f"{i}" for i in range(29)] + ["NA"])
        if_unexpected_values(X, "RA_alert", RA_alert)
        #"Hosp_dato_slut" - set to 31DEC9999 if continued
        #"Lag_diagnosis_date_con" - 0 and 1
#        X['patient_id'] = X['patient_id'].astype(str).str.strip()
#        for patient in X['patient_id'].unique():
#            print(patient)
#            subset = X[X['patient_id'] == patient]
#            interventions = subset['Forloebs_id'].unique()
#            if len(interventions) > 1:
#                print(f"Warning: Multiple interventions found for patient {patient}. Excluding the patient due to multiple diagnosis types.")
#                patients_multiple_diagnosis.add(patient)

#maybe filter out later when creating additional files
    if dataset in ["patients", "treatments", "visits", "yearly_visits", "saes", "logistics"]:
        for patient in X['patient_id'].unique():
            if patient in patients_wrong_diagnosis:
                continue
            if patient in patients_multiple_diagnosis:
                continue
            subset = X[X['patient_id'] == patient]
            diagnosis = subset['Diagnosis'].unique()
            if not np.isin(diagnosis, RA_diagnosis).all():
                patients_wrong_diagnosis.add(patient)
                print(f"Warning: Wrong diagnosis found for patient {patient}. Excluding the patient.")
                print(diagnosis)
            elif len(diagnosis) > 1:
                print(f"Warning: Multiple diagnoses found for patient {patient}. Excluding the patient due to multiple diagnosis types.")
                print(diagnosis)
                patients_multiple_diagnosis.add(patient)
#            if not np.isin(diagnosis, RA_diagnosis).all():
#                patients_wrong_diagnosis.add(patient)

#save to the dictionary
    filtered_datasets[dataset] = X

print("////////////////////////////////////////////////////////////////////////////")
print("DONE FILTERING, MOVING ON TO EXCLUDING WRONG OR MULTIPLE DIAGNOSIS PATIENTS")
print("WRONG DIAGNOSIS PATIENTS:")
print(len(patients_wrong_diagnosis))
print(patients_wrong_diagnosis)
with open('../../data/excluded/patients_wrong_diagnosis.txt', 'w') as f:
    f.writelines(f"{patient}\n" for patient in sorted(patients_wrong_diagnosis))
print("MULTIPLE DIAGNOSIS PATIENTS:")
print(len(patients_multiple_diagnosis))
print(patients_multiple_diagnosis)
with open('../../data/excluded/patients_multiple_diagnosis.txt', 'w') as f:
    f.writelines(f"{patient}\n" for patient in sorted(patients_multiple_diagnosis))
for dataset_name, dataset in filtered_datasets.items():
        print(f"Excluded entries from dataset: {dataset_name}")
        #diagnosis
        print("MULTIPLE DIAGNOSIS")
        excluded_multiple = dataset[dataset["patient_id"].isin(patients_multiple_diagnosis)]
        print(excluded_multiple)
        excluded_multiple.to_csv(f'../../data/excluded/{dataset_name}/{dataset_name}_multiple_diag_excluded.csv', index=False)
        dataset = dataset[~dataset["patient_id"].isin(patients_multiple_diagnosis)]
        print("WRONG DIAGNOSIS")
        excluded_wrong = dataset[dataset["patient_id"].isin(patients_wrong_diagnosis)]
        print(excluded_wrong)
        excluded_wrong.to_csv(f'../../data/excluded/{dataset_name}/{dataset_name}_wrong_diag_excluded.csv', index=False)
        dataset = dataset[~dataset["patient_id"].isin(patients_wrong_diagnosis)]

#save filtered patients dataset
        dataset.to_csv(f'../data/{dataset_name}_filtered.csv', index=False)