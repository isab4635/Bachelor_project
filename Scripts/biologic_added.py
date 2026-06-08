#!/usr/local/anaconda3-2024.10-1/bin/python3
# The purpose of this script is to get a file with patient_id and biologic_added (0: no, 1: yes)
# This is done on the basis of a list of patients with drugs added within first year and
# final_timeline_csv_after_all_exclusions.csv for all the patients

# Import libraries
import pandas as pd
import numpy as np

# Load in timeline and lists
path = "../data/"
df = pd.read_csv(path + 'final_timeline_csv_after_all_exclusions.csv')

with open("../data/list_patients_on_bio.txt") as f:
    patients_on_bio = [line.strip() for line in f if line.strip()]

with open("../data/list_patients_on_bio_and_ts.txt") as f:
    patients_on_bio_and_ts = [line.strip() for line in f if line.strip()]

with open("../data/list_patients_on_ts.txt") as f:
    patients_on_ts = [line.strip() for line in f if line.strip()]

# Merge patients that had biologics added with patients on ts_dmard or both
patients_on_bio.extend(patients_on_bio_and_ts)
patients_on_bio.extend(patients_on_ts)

# Print patients at start
count = df["patient_id"].nunique()
print("Number of patients: ")
print(count)

# Initialize dataframe with one row per patient
unique_ids = df["patient_id"].unique()
biologic_added = pd.DataFrame({"patient_id": unique_ids, "biologic_added": 0})

# Define the label
biologic_added['biologic_added'] = biologic_added['patient_id'].isin(patients_on_bio).astype(int)

# Attach start_mtx to all rows of that patient
mtx_dates = df[['patient_id','first_mtx_date']].groupby('patient_id').first().reset_index()
biologic_added = biologic_added.merge(mtx_dates, on='patient_id', how='left')

# Count number of patients on biologic and not on biologic
subset_bio = biologic_added[biologic_added['biologic_added'] == 1]
print("Number of patients on biologic: ")
print(subset_bio.shape[0])
subset_no_bio = biologic_added[biologic_added['biologic_added'] == 0]
print("Number of patients not on biologic: ")
print(subset_no_bio.shape[0])

# Check counts together with nan values
nan_included_counts = biologic_added['biologic_added'].value_counts(dropna=False)
print(nan_included_counts)

# ---- Save file ---- #
biologic_added.to_csv(path + "biologic_added.csv", index=False)