#!/usr/local/anaconda3-2024.10-1/bin/python3
import pandas as pd

df = pd.read_csv('../data/timeline_mtx_normalized.csv')
X = df.copy()

# Check how many missing diagnosis date and drop them
count_no_con = X['diagnosis_date'].isna().sum()
X = X.dropna()
X['diagnosis_date'] = pd.to_datetime(X['diagnosis_date'], format = "mixed")

# Seperation
df_before_2015 = X[X['diagnosis_date'] < pd.to_datetime('2015-01-01')]
df_after_2015 = X[X['diagnosis_date'] >= pd.to_datetime('2015-01-01')]

print(f'number of missing constructed date: {count_no_con}')
print(f'number of patients diagnosis before: {df_before_2015['patient_id'].nunique()}')
print(f'number of patients diagnosis after: {df_after_2015['patient_id'].nunique()}')

#save seperated patients dataset
df_before_2015.to_csv(f'../data/timeline_mtx_normalized_before_2015.csv', index=False)
df_after_2015.to_csv(f'../data/timeline_mtx_normalized_after_2015.csv', index=False)
