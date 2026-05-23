#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 6/5/2026
# The purpose of this script is to get the data as a feature table with one row per patient
# Info from visits should be date closest to mtx_start
# No missingness

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load in mtx_status as a start
path = "../data/"
df_status = pd.read_csv(path + 'mtx_status.csv')
df_status['first_mtx_date'] = pd.to_datetime(df_status['first_mtx_date'])

print(f"Number of patients at start: {df_status["patient_id"].nunique()}")


# ---- Extract diagnosis and age_at_diagnosis from patient_filtered.csv ---- #
df_patients = pd.read_csv(path + 'patients_filtered.csv', usecols=['patient_id', 'Diagnosis', 'Age_at_diagnosis'])
# Limit to just one row per patient and merge
df_patients = df_patients.groupby('patient_id').first().reset_index()
df_status = df_status.merge(df_patients, on='patient_id', how='left')

print(f"Number of patients after adding patient info: {df_status["patient_id"].nunique()}")


# ---- Extract patient sex from logistics ---- #
df_logistics = pd.read_csv(path + 'logistics_filtered.csv', usecols=['patient_id', 'Koen'])
# Limit to just one row per patient and merge
df_logistics = df_logistics.drop_duplicates()
df_status = df_status.merge(df_logistics, on='patient_id', how='left')

print(f"Number of patients after adding sex: {df_status["patient_id"].nunique()}")

# ---- Column for whether biologic was added within first year ---- #
df_biologic = pd.read_csv(path + 'biologic_added.csv', usecols=['patient_id', 'biologic_added']).drop_duplicates()
df_status = df_status.merge(df_biologic, on='patient_id', how='left')

print(f"Number of patients after adding biologic_added: {df_status["patient_id"].nunique()}")


# ---- Extract info of antibodies ---- #
# Extract patient igm from logistics
igm = pd.read_csv(path +'logistics_filtered.csv', usecols=['patient_id', 'IgM_RF', 'Diagnosis_date_con', 'IgM_RF_aar']).drop_duplicates()
# Find the igm year closest to diagnosis date and merge
X = igm.copy()
X = X[X['patient_id'].isin(df_status['patient_id'])]
X['Diagnosis_date_con'] = pd.to_datetime(X['Diagnosis_date_con'])
X['igm_normalized_date'] = (X['IgM_RF_aar'] - X['Diagnosis_date_con'].dt.year).abs()

X = X.sort_values(by = 'igm_normalized_date')
X = X.drop_duplicates(subset=['patient_id'])
X = X.drop(columns = ['Diagnosis_date_con', 'IgM_RF_aar', 'igm_normalized_date'])

df_status = df_status.merge(X, on='patient_id', how='left')

print(f"Number of patients after adding igm: {df_status["patient_id"].nunique()}")

# Extract patient accp from logistics
accp = pd.read_csv(path +'logistics_filtered.csv', usecols=['patient_id', 'Anti_CCP', 'Diagnosis_date_con', 'Anti_CCP_aar'])
# Find the accp year closest to diagnosis date and merge
X = accp.copy()
X = X[X['patient_id'].isin(df_status['patient_id'])]
X['Diagnosis_date_con'] = pd.to_datetime(X['Diagnosis_date_con'])
X['anti_ccp_normalized_date'] = (X['Anti_CCP_aar'] - X['Diagnosis_date_con'].dt.year).abs()

X = X.sort_values(by = 'anti_ccp_normalized_date')
X = X.drop_duplicates(subset=['patient_id'])
X = X.drop(columns = ['Diagnosis_date_con', 'Anti_CCP_aar', 'anti_ccp_normalized_date'])

df_status = df_status.merge(X, on='patient_id', how='left')

print(f"Number of patients after adding anti-ccp: {df_status["patient_id"].nunique()}")

# Map the categories to 0 and 1
df_status["IgM_RF"] = df_status["IgM_RF"].map({"Neg" : 0, "Pos" : 1})
df_status["Anti_CCP"] = df_status["Anti_CCP"].map({"Neg" : 0, "Pos" : 1})


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
    direction='backward',
    tolerance=pd.Timedelta('90D')
)

# ---- Finalize and save feature table ---- #
# Calculate MDHAQ if missing using Haq (MDHAQ = (Haq - 0.15) / 1.08)
df_status['cal_MDHAQ'] = (df_status['Haq'] - 0.15) / 1.08
df_status['MDHAQ'] = df_status['MDHAQ'].fillna(df_status['cal_MDHAQ'])
df_status = df_status.drop(columns = ['Haq', 'cal_MDHAQ'])

# Drop columns that are not features or labels
df_status = df_status.drop(columns=["patient_id", "Diagnosis", "first_mtx_date", "Visit_date"])
# Drop NAs
df_status.dropna(inplace=True)

print("No NAs:", len(df_status.index))

# Save to csv
df_status.to_csv(path + "filled_feature_table.csv", index=False)