#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 18/3/2026

# Import libraries
import pandas as pd

path = "../data/"

# Read in the visits.csv
df_visits = pd.read_csv(path + "visits_filtered.csv", encoding= 'unicode_escape')

df_long = df_visits.melt(
    id_vars = ["patient_id", "Visit_date"],
    var_name = "Event",
    value_name = "Value"
).dropna()

df_long.rename(columns={'Visit_date': 'Date'}, inplace=True)

del df_visits


# Read in treatments.csv
df_treat = pd.read_csv(path + "treatments_filtered.csv", encoding= 'unicode_escape', usecols=["Prescription","Prescription_start_date","Prescription_stop_date","patient_id"])

df_treat["Prescription"] = df_treat["Prescription"].astype("category")

# Treatment start df
df_start = df_treat[["patient_id", "Prescription_start_date", "Prescription"]].copy().dropna()
df_start.columns = ["patient_id", "Date", "Value"]
df_start["Event"] = "Prescription_start"

# Treatment stop df
df_stop = df_treat[["patient_id", "Prescription_stop_date", "Prescription"]].copy().dropna()
df_stop.columns = ["patient_id", "Date", "Value"]
df_stop["Event"] = "Prescription_stop"

del df_treat


# Read in yearly_visits.csv
df_yr_visits = pd.read_csv(path + "yearly_visits_filtered.csv", encoding= 'unicode_escape')
na_rows = df_yr_visits["Year_of_status"].isna()
df_yr_visits = df_yr_visits[~na_rows]
df_yr_visits["Year_of_status"] = df_yr_visits["Year_of_status"].astype(int).astype(str) + "-01-01"

df_yr_long = df_yr_visits.melt(
    id_vars = ["patient_id", "Year_of_status"],
    var_name = "Event",
    value_name = "Value"
).dropna()

df_yr_long.rename(columns={'Year_of_status': 'Date'}, inplace=True)

del df_yr_visits


# Read in saes.csv
df_saes = pd.read_csv(path + "saes_filtered.csv",
                       encoding= 'unicode_escape',
                       usecols=["patient_id","Sae_start_date","Sae_stop_date","Sae_number"])

# Sae start df
df_sae_start = df_saes[["patient_id", "Sae_start_date", "Sae_number"]].copy().dropna()
df_sae_start.columns = ["patient_id", "Date", "Value"]
df_sae_start["Event"] = "Sae_start"

# Sae stop df
df_sae_stop = df_saes[["patient_id", "Sae_stop_date", "Sae_number"]].copy().dropna()
df_sae_stop.columns = ["patient_id", "Date", "Value"]
df_sae_stop["Event"] = "Sae_stop"

# Combine all the dataframes
df_long = pd.concat([df_long, df_start, df_stop, df_yr_long, df_sae_start, df_sae_stop], ignore_index=True)

# Sort them according to patient, then date, then event
df_long.sort_values(by=["patient_id", "Date"], inplace=True)

df = df_long
df['Date'] = pd.to_datetime(df['Date'], format = 'mixed')

# ---- Normalize to MTX ---- #
# Filter to mtx start events
mtx_starts = df[(df['Event'] == 'Prescription_start') & (df['Value'].isin(['DMARD_MTX_SC','DMARD_MTX', 'DMARD_MTX_IM']))]

# First time each patient took mtx
first_mtx = (mtx_starts.groupby('patient_id')['Date'].min().rename('first_mtx_date').reset_index())

# Attach first_mtx_date to all rows of that patient
df_mtx = df.copy().merge(first_mtx, on='patient_id', how='left')
count = df_mtx["patient_id"].nunique()
print("Number of patients before dropping due to mtx start missing: ")
print(count)

# Normalized time: months since first MTX
df_mtx['months_since_mtx'] = (df_mtx['Date'] - df_mtx['first_mtx_date']) / np.timedelta64(1, 'D')/30.44
#df_mtx = df_mtx.drop('first_mtx_date', axis = 1)

# Drop those with no first_mtx_date
df_mtx.loc[df_mtx['months_since_mtx'].isnull()].to_csv("../data/normalized_excluded/nan_mtx_date_after_2015.csv")
df_mtx = df_mtx.dropna(subset=['months_since_mtx'])

#print the amount of patients dropped due to MTX prescription missing
count = df_mtx["patient_id"].nunique()
print("Number of patients after the drop: ")
print(count)

# Get diagnosis dates
df_diag_date = df[df['Event'] == 'Diagnosis_date_con'].copy()
df_diag_date['Value'] = pd.to_datetime(df_diag_date['Value'])

# Earliest diagnosis date for each patient
first_diag = df_diag_date.groupby('patient_id')['Value'].min().rename('diagnosis_date').reset_index()
# Attach diagnosis date to all rows of that patient
df_mtx = df_mtx.merge(first_diag, on='patient_id', how='left')

#print the final amount of patients
count = df_mtx["patient_id"].nunique()
print("Number of patients after adding diagnosis: ")
print(count)

# Save to csv
df_mtx.to_csv('../data/timeline_mtx_normalized.csv', index=False)