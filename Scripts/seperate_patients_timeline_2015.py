#!/usr/local/anaconda3-2024.10-1/bin/python3
import pandas as pd
import numpy as np

df = pd.read_csv('../data/timeline_style.csv')
X = df.copy()

patients_before_2015 = set()
patients_after_2015 = set()
excluded_no_diagnosis_date = []
count_no_con = 0

for patient in X['patient_id'].unique():

#creating subsets to adjust them
        subset = X[X['patient_id'] == patient]
        Diagnosis = subset[subset['Event'] == 'Diagnosis_date_con']

#checking if diagnosis date missing
        if Diagnosis.empty:
                print("............................................")
                print(f"No constructed date found - patient won't be included in the model")
                count_no_con += 1
                excluded_no_diagnosis_date.append(patient)
                continue

#taking the earliest diagnosis date
        Diagnosis_date = pd.to_datetime(Diagnosis.iloc[0,3], format = "mixed")
        if Diagnosis_date < pd.to_datetime('2015-01-01'):
                patients_before_2015.add(patient)
        else:
                patients_after_2015.add(patient)

df_before_2015 = X[X['patient_id'].isin(patients_before_2015)]
df_after_2015 = X[X['patient_id'].isin(patients_after_2015)]

print(f'number of missing constructed date: {count_no_con}')

#save seperated patients dataset
df_before_2015.to_csv(f'../data/timeline_diagnosis_before_2015.csv', index=False)
df_after_2015.to_csv(f'../data/timeline_diagnosis_after_2015.csv', index=False)
with open(f'../data/excluded/timeline_no_diagnosis_date.txt', 'w') as f:
        for line in excluded_no_diagnosis_date:
                f.write(f"{line}\n")

