#!/usr/local/anaconda3-2024.10-1/bin/python3
# To compare two entries: diagnosis_date and constructed diagnosis date, to choose most suitable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
X = pd.read_csv('../data/timeline_style.csv')

df_1 = X[X['Event']=='Diagnosis_date']
df_2 = X[X['Event']=='Diagnosis_date_con']

df_1 = df_1.groupby('patient_id')['Value'].first().reset_index()
df_2 = df_2.groupby('patient_id')['Value'].first().reset_index()

# Compare the two diagnosis dates per patient
compare = df_1.merge(df_2, on='patient_id', suffixes=('_original', '_constructed'))
compare['is_the_same'] = compare['Value_original'] == compare['Value_constructed']

# Analyze results and one of the patients with wrongly constructed dates
wrong_con = compare[compare['is_the_same'] == False]
print(len(wrong_con))
print(wrong_con.head())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

suspicious_patient = X[X['patient_id'] == wrong_con.iloc[4]['patient_id']]
suspicious_patient.to_csv(f'../data/suspicious_patient.csv')

# Save the patients with wrong constructed diagnosis date for further analysis
wrong_con.to_csv(f'../data/wrong_constructed_diagnosis_date_patients.csv')

# Check the time between each date and first prescription
df_3 = X[X['Event'] == 'Prescription_start']
df_3 = df_3.groupby('patient_id')['Date'].min().reset_index()
df_3 = df_3[df_3['patient_id'].isin(wrong_con['patient_id'])]

compare_wrong = wrong_con.merge(df_3, on='patient_id', suffixes=('_from_comparison', '_from_prescription'))
print(compare_wrong.head())

compare_wrong['Date'] = pd.to_datetime(compare_wrong['Date'])
compare_wrong['Value_original'] = pd.to_datetime(compare_wrong['Value_original'])
compare_wrong['Value_constructed'] = pd.to_datetime(compare_wrong['Value_constructed'])

compare_wrong['prescription_since_original_D'] = ((compare_wrong['Date'] - compare_wrong['Value_original']) / np.timedelta64(1, 'D'))
compare_wrong['prescription_since_constructed_D'] = ((compare_wrong['Date'] - compare_wrong['Value_constructed']) / np.timedelta64(1, 'D'))

# Add decision column
compare_wrong['which_closer'] = compare_wrong[['prescription_since_original_D', 'prescription_since_constructed_D']].abs().idxmin(axis=1)

print(compare_wrong.head())
compare_wrong.to_csv(f'../data/compare_wrong_constructed_prescriptions.csv')

# Plot and save the results
count = compare_wrong['which_closer'].value_counts()
count.plot.bar()
plt.title('Compare two diagnosis date entries - which one closer to first prescription')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.savefig(f'../data/figures/diagnosis_date_org_vs_con.png', dpi=300, bbox_inches='tight')