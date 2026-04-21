#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 14/4/2026
# The purpose of this script is to get a file with patient_id and mtx_fail (0: fail, 1: success)
# This is done on the basis of a timeline style file (normalized to mtx and with diagnosis date)


# Import libraries
import pandas as pd
import numpy as np

# Load in timeline
path = "../data/"
df = pd.read_csv(path + 'timeline_mtx_normalized_after_2015.csv')
# Ensure dates are in datetime
df['Date'] = pd.to_datetime(df['Date'])
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['first_mtx_date'] = pd.to_datetime(df['first_mtx_date'])

# ---- Select what patients we want to look at when determining mtx_fail ---- #

# Print patients at start
count = df["patient_id"].nunique()
print("Number of patients before anything: ")
print(count)

### Ensure that patients do have a check-in visit some time after mtx_start ###
# Find max visit date. (Inspired by https://stackoverflow.com/questions/23394476/keep-other-columns-when-doing-groupby)
max_visit_dates = df.loc[df.groupby("patient_id")["Date"].idxmax()]
patients_no_follow_up = max_visit_dates[max_visit_dates["Date"] <= max_visit_dates["first_mtx_date"]]

patients_no_follow_up.to_csv("../data/filtered_mtx_fail/patients_no_follow_up.csv")
df = df[~df["patient_id"].isin(patients_no_follow_up)]

# Print number of patients after removing those with no check-in
count = df["patient_id"].nunique()
print("Number of patients after ensuring follow-up visit: ")
print(count)


### Ensure that mtx_start is close to diagnosis_date (within 6 months) ###
# Limit to just one row per patient (as diag_date and first_mtx_date is date-independent)
temp_date_df = df.groupby('patient_id').first().reset_index()

# Find gap between diagnosis_date and first_mtx date
temp_date_df['date_diff_days'] = (temp_date_df['diagnosis_date'] - temp_date_df['first_mtx_date']) / np.timedelta64(1, 'D')
# Filter patients where absolute gap/difference is more than 6 months
patients_mtx_diag_gap = temp_date_df[abs(temp_date_df['date_diff_days']) > 30.44*6]["patient_id"]

df[df['patient_id'].isin(patients_mtx_diag_gap)].to_csv("../data/filtered_mtx_fail/patients_mtx_diag_gap.csv")
df = df[~df['patient_id'].isin(patients_mtx_diag_gap)]

# Print patients after removing those with more than 6 months between diag_date and start
count = df["patient_id"].nunique()
print("Number of patients with less than 6 months between diag and mtx: ")
print(count)


# ---- Determine if MTX fail in first year --- #
# Failure: Prescription stop of mtx lasts more than 3 months. Value of 0 in mtx_fail
# Limit to only first year (12+3 months to test if they get back on mtx)
df = df[(df['months_since_mtx'] <= 15) & (df['months_since_mtx'] >= 0)]

# Get all MTX events
mtx_events = df[df['Value'].isin(['DMARD_MTX_SC','DMARD_MTX', 'DMARD_MTX_IM'])].copy()

# Shift to make new column with next event and its date
mtx_events["Next_event"] = mtx_events.groupby("patient_id")["Event"].shift(-1)
mtx_events["Next_date"] = mtx_events.groupby("patient_id")["Date"].shift(-1)

# Add column with time between events (gap)
mtx_events["Gap"] = mtx_events["Next_date"] - mtx_events["Date"]

# Initialize dataframe with one row per patient to collect mtx_failures
unique_ids = df["patient_id"].unique()
mtx_status = pd.DataFrame({"patient_id": unique_ids, "mtx_fail": None})

# Go over each mtx_stop event
for stops in mtx_events[mtx_events["Event"] == "Prescription_stop"].itertuples():
    # If the gap is less than 90 days and next event is a prescription start, move on
    if stops.Gap <= pd.Timedelta(days = 30.44*3) and stops.Next_event == "Prescription_start":
        continue
    # Else this stop indicates an mtx failure and is saved as such
    else:
        mtx_status.loc[mtx_status["patient_id"] == stops.patient_id, "mtx_fail"] = 0

# Every value in mtx_fail which is still none is a success and set to 1
mtx_status['mtx_fail'] = mtx_status['mtx_fail'].fillna(1).astype(int)


# ---- Save file ---- #
# Print info about it
print(f"Total patients: {len(mtx_status.index)}")
nan_included_counts = mtx_status['mtx_fail'].value_counts(dropna=False)
print(nan_included_counts)

mtx_status.to_csv("../data/mtx_status.csv")