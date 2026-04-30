#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 30/4/2026
# The purpose of this script is to get the data as a feature table with one row per patient
# Info from visits should be date closest to mtx_start

# Import libraries
import pandas as pd
import numpy as np

# Load in mtx_status as a start
path = "../../data/"
df_status = pd.read_csv(path + 'mtx_status.csv')
df_status['first_mtx_date'] = pd.to_datetime(df_status['first_mtx_date'])

print(f"Number of patients at start: {df_status["patient_id"].nunique()}")


# ---- Extract diagnosis and age_at_diagnosis from patients_filtered.csv ---- #
df_patients = pd.read_csv(path + 'patients_filtered.csv', usecols=['patient_id', 'Diagnosis', 'Age_at_diagnosis'])
# Limit to just one row per patient
df_patients = df_patients.groupby('patient_id').first().reset_index()

df_status = df_status.merge(df_patients, on='patient_id', how='left')

print(f"Number of patients after adding patient info: {df_status["patient_id"].nunique()}")

# ---- Extract patient sex from logistics ---- #
df_logistics = pd.read_csv(path + 'logistics_filtered.csv', usecols=['patient_id', 'Koen'])
# Limit to just one row per patient
df_logistics = df_logistics.drop_duplicates()

df_status = df_status.merge(df_logistics, on='patient_id', how='left')

print(f"Number of patients after adding sex: {df_status["patient_id"].nunique()}")


# ---- Extract info from visits ---- #
visits_selected_vars = ['patient_id', 'Visit_date', 'CRP', 'Haq', 'MDHAQ', 'DAS_28_CRP', 'Vas_patient_global',
                        'Vas_patient_pain', 'Vas_patient_fatigue', 'Vas_doctor', 'CDAI', 'SDAI', 'Swollenjoints28']
df_visits = pd.read_csv(path + 'visits_filtered.csv', usecols=visits_selected_vars)
df_visits['Visit_date'] = pd.to_datetime(df_visits['Visit_date'])

# Ensure sorted
df_status = df_status.sort_values(['first_mtx_date', 'patient_id'])
df_visits = df_visits.sort_values(['Visit_date', 'patient_id'])

# Merge asof allows for merging on nearest key
df_status = pd.merge_asof(
    df_status,
    df_visits,
    left_on='first_mtx_date',
    right_on='Visit_date',
    by='patient_id',
    direction='nearest',
    tolerance=pd.Timedelta('60D')
)

# Drop date columns
df_status = df_status.drop(columns=["first_mtx_date", "Visit_date"])

# Save to csv
df_status.to_csv("feature_table_attempt_60D.csv", index=False)