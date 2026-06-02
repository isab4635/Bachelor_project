#!/usr/local/anaconda3-2024.10-1/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


df = pd.read_csv('../data/timeline_diagnosis_after_2015.csv')
X = df.copy()

X_cdai = X[X['Event'] == 'CDAI']

x = 0
fig, ax = plt.subplots(constrained_layout=True)
mtx_dictionary = {'DMARD_MTX': 'MTX', 'DMARD_MTX_SC': 'MTX', 'DMARD_MTX_IM': 'MTX'}

for patient in X_cdai['patient_id'].unique()[:100]:

#creating subsets to adjust them
        subset = X[X['patient_id'] == patient]
        subset_cdai = X_cdai[X_cdai['patient_id'] == patient]
        subset_prescription_start = subset[subset['Event'] == 'Prescription_start']
        subset_prescription_stop = subset[subset['Event'] == 'Prescription_stop']

#ignoring different admininstration methods for now
        subset_prescription_start['Value'] = subset_prescription_start['Value'].replace(mtx_dictionary)
        subset_prescription_stop['Value'] = subset_prescription_stop['Value'].replace(mtx_dictionary)

#Finding only MTX start values
        MTX_start = subset_prescription_start[subset_prescription_start['Value'].isin(['MTX'])]
        MTX_stop = subset_prescription_stop[subset_prescription_stop['Value'].isin(['MTX'])]

#checking if CDAI missing or MTX missing
        if MTX_start.empty or subset_cdai.empty:
                print(f"Missing CDAI or prescrption start date values for patient {patient}.")
                x+=1
                continue

        if not MTX_stop.empty:
                MTX_stop = MTX_stop.reset_index(drop=True)
                starts = pd.to_datetime(MTX_start['Date'], format = "%Y-%m-%d")
                stops = pd.to_datetime(MTX_stop['Date'], format = "%Y-%m-%d")
                differences = (stops.values[:, None] - starts.values) / np.timedelta64(1, 'D')
                mask = ~((differences >= 0) & (differences <= 90)).any(axis=1)
                MTX_stop = MTX_stop[mask]
                print(f"Excluded the break in prescription smaller than 3 months for patient {patient}")

#focusing only on stops during the first year
                if not MTX_stop.empty:
                       MTX_stop = MTX_stop.reset_index(drop=True)
                       starts = pd.to_datetime(MTX_start['Date'], format = "%Y-%m-%d")
                       start = starts.iloc[0]
                       first_year = start + pd.DateOffset(months=12)
                       MTX_stop = MTX_stop[pd.to_datetime(MTX_stop['Date']) <= first_year]

#checking if more than one diagnosis date
        if len(MTX_start) > 1:
                print(f"Warning: Multiple prescription starts found for patient {patient}. Using the first one for MTX.")
        #       MTX_start = pd.to_datetime(MTX_start, format = "%Y-%m-%d")
        #       MTX_start = MTX_start.sort_values(by='date', ascending=True)
        #       print(MTX_start)
                MTX_start = MTX_start.iloc[0:1]
#normalizing
        MTX_date = MTX_start.iloc[:, 1]
        MTX_date = pd.to_datetime(MTX_date, format = "%Y-%m-%d")
        x_dates = (pd.to_datetime(subset_cdai.iloc[:, 1], format = "%Y-%m-%d").values - MTX_date.values) / np.timedelta64(1, 'D')/30.44

        if MTX_stop['Value'].empty:
                colours = 'green'
        else:
                colours = 'red'
#plotting
        ax.plot(x_dates,
                subset_cdai.iloc[:, 3].values.astype(float),
                label=patient, alpha=0.5, marker = 'o', color = colours)
custom_lines = [Line2D([0], [0], color='green', lw=4),
                Line2D([0], [0], color='red', lw=4)]
ax.legend(custom_lines, ['MTX continued', 'MTX stopped'])
ax.set_title('CDAI over time normalized to MTX start date (months) - diagnosis after 2015')
plt.xlim(-1,13)
plt.xticks(rotation=90)
plt.show()
print(f"Number of patients missing CDAI or MTX prescription: {x}")