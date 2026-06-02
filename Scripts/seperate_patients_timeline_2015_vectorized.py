#!/usr/local/anaconda3-2024.10-1/bin/python3
# This script separates patients into two groups based on their diagnosis date: those diagnosed before 2015 and 
# those diagnosed in 2015 or later. It reads in the timeline dataset and the logistics dataset, 
# identifies patients with missing diagnosis dates, and saves the separated datasets for further analysis.

# Import necessary libraries
import pandas as pd

# Read in the timeline dataset and the logistics dataset
df = pd.read_csv('../data/timeline_style.csv')
X_full = df.copy()
df = pd.read_csv('../data/logistics_filtered.csv', usecols = ['patient_id', 'Diagnosis_date_con']).drop_duplicates()
X = df.copy()

# Initialize sets to store patient IDs for those diagnosed before and after 2015
patients_before_2015 = set()
patients_after_2015 = set()
count_no_con = X['Diagnosis_date_con'].isna().sum()
X = X.dropna()
X['Diagnosis_date_con'] = pd.to_datetime(X['Diagnosis_date_con'], format = "mixed")

#Seperate
patients_before_2015 = X[X['Diagnosis_date_con'] < pd.to_datetime('2015-01-01')]
patients_after_2015 = X[X['Diagnosis_date_con'] >= pd.to_datetime('2015-01-01')]
df_before_2015 = X_full[X_full['patient_id'].isin(patients_before_2015['patient_id'])]
df_after_2015 = X_full[X_full['patient_id'].isin(patients_after_2015['patient_id'])]

print(f'number of missing constructed date: {count_no_con}')
print(f'number of patients diagnosis before: {df_before_2015['patient_id'].nunique()}')
print(f'number of patients diagnosis after: {df_after_2015['patient_id'].nunique()}')

#save seperated patients dataset
df_before_2015.to_csv(f'../data/timeline_diagnosis_before_2015.csv', index=False)
df_after_2015.to_csv(f'../data/timeline_diagnosis_after_2015.csv', index=False)