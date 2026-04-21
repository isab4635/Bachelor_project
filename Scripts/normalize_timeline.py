#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 14/04/2026
# Import packages
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('../data/timeline_diagnosis_after_2015.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Number of patients at start
count = df["patient_id"].nunique()
print("Number of patients before droppings: ")
print(count)

# ---- Normalize to MTX ---- #
# Filter to mtx start events
mtx_starts = df[(df['Event'] == 'Prescription_start') & (df['Value'].isin(['DMARD_MTX_SC','DMARD_MTX', 'DMARD_MTX_IM']))]

# First time each patient took mtx
first_mtx = mtx_starts.groupby('patient_id')['Date'].min().rename('first_mtx_date').reset_index()

# Attach first_mtx_date to all rows of that patient
df_mtx = df.copy().merge(first_mtx, on='patient_id', how='left')

# Normalized time: months since first MTX
df_mtx['months_since_mtx'] = (df_mtx['Date'] - df_mtx['first_mtx_date']) / np.timedelta64(1, 'D')/30.44

# Drop those with no first_mtx_date
df_mtx = df_mtx.dropna(subset=['months_since_mtx'])

# Print the amount of patients after dropped due to MTX prescription missing
count = df_mtx["patient_id"].nunique()
print("Number of patients with MTX start presctiption: ")
print(count)


# ---- Add diagnosis date as its own column ---- #
# Get diagnosis dates
df_diag_date = df[df['Event'] == 'Diagnosis_date'].copy()
df_diag_date['Value'] = pd.to_datetime(df_diag_date['Value'])

# Earliest diagnosis date for each patient
first_diag = df_diag_date.groupby('patient_id')['Value'].min().rename('diagnosis_date').reset_index()

# Attach diagnosis date to all rows of that patient
df_mtx = df_mtx.merge(first_diag, on='patient_id', how='left')


# ---- Save to csv ---- #
df_mtx.to_csv('../data/timeline_mtx_normalized_after_2015.csv')
