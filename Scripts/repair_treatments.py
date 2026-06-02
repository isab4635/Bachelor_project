#!/usr/local/anaconda3-2024.10-1/bin/python3
# To exclude error prescriptions that have been accidently started and stopped on the same day
import pandas as pd

# Read in treatments.csv
df_treat = pd.read_csv("../data/treatments_filtered.csv", encoding= 'unicode_escape')
mask = df_treat['Prescription_start_date'] == df_treat['Prescription_stop_date']
print(f'original number of prescriptions: {len(df_treat)}')
df_wrong = df_treat[mask]
print(f'number of prescriptions same day start and stop: {len(df_wrong)}')
df = df_treat[~mask]
df.to_csv("../data/treatments_filtered_final.csv", index = False)