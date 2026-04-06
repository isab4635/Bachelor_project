#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 06/04/2026

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Define path where files should be saved to
to_path = "../results/desc_stats/"

# Import timeline
timeline_df = pd.read_csv('../data/timeline_style.csv')


# ---- Most prevalent variables ---- #
# Load dataframe
events = timeline_df['Event'].copy()

# Get top 10 events
counts = events.value_counts(sort = True)

# !!!Hard code!!!, not looking at first 9 as they are basic info
counts_head = counts.iloc[9:].head(10)

# Horizontal bar plot for top 10
plt.figure(figsize=(8, 5))
ax = counts_head[::-1].plot.barh()
# Add value labels
for i, v in enumerate(counts_head[::-1]):
	ax.text(v + 0.5, i, str(v), va='center')
# Plot design
plt.ylabel("Event")
plt.xlabel("Entries")
plt.title("Top 10 most prevalent measurement events")
plt.tight_layout()
plt.savefig(to_path + "top_10_events.png")

# Horizontal bar plot for all
n = len(counts)
plt.figure(figsize=(10, 0.3 * n))
ax = counts[::-1].plot.barh()
# Add value labels
for i, v in enumerate(counts[::-1]):
	ax.text(v + 0.5, i, str(v), va='center')
# Plot design
plt.ylabel("Event")
plt.xlabel("Entries")
plt.title("Events in timeline")
plt.tight_layout()
plt.savefig(to_path + "count_events.png")


# ---- Histograms over most prevalent variables ---- #
# !!!Hard code!!!
vars = ['CRP', 'Vas_patient_global', 'Vas_patient_pain', 'Vas_patient_fatigue', 'DAS_28_CRP', 'Vas_doctor', 'CDAI', 'SDAI', 'DAPSA','Haq', 'MDHAQ']

# Load timeline_style.csv and extract values of chosen variable
df_event_val = timeline_df[['Event', 'Value']].copy()

# Plotting histogram
fig, axs = plt.subplots(5, 2)

for i in range(5):
	df_var_values_1 = df_event_val[df_event_val['Event'] == vars[i]]['Value']
	df_var_values_1 = df_var_values_1.str.replace(',', '.').astype(float)

	df_var_values_2 = df_event_val[df_event_val['Event'] == vars[i*2]]['Value']
	df_var_values_2 = df_var_values_2.str.replace(',', '.').astype(float)

	axs[i][0].hist(df_var_values_1, bins=12, edgecolor='black', color='red', alpha=0.7, density=True)
	axs[i][1].hist(df_var_values_2, bins=12, edgecolor='black', color='red', alpha=0.7, density=True)
	axs[i][0].set_title(f'{vars[i]} ({len(df_var_values_1)} entries)')
	axs[i][1].set_title(f'{vars[i*2]} ({len(df_var_values_2)} entries)')

fig.suptitle("Histograms 12 bins")
plt.tight_layout()
plt.savefig(to_path + "histograms_top_events.png")

# Individual histograms with maximum highlighted
for var in vars:
	df_var_vals = df_event_val[df_event_val['Event'] == var]['Value'].astype(float)

	plt.figure()
	plt.hist(df_var_vals, bins=20, edgecolor='black', color='red', alpha=0.7)
	plt.title(f'{var} ({len(df_var_vals)} entries), 20 bins')

	plt.axvline(max(df_var_vals), linestyle='dotted', linewidth=2)
	plt.text(max(df_var_vals), plt.ylim()[1] * 0.9, f'Max: {max(df_var_vals)}')

	plt.tight_layout()
	plt.savefig(f"{to_path}histogram_{var}.png")


# ---- Visits per patient ---- #
# Load visits.csv
df_visits = pd.read_csv('../data/visits_filtered.csv', usecols = ['patient_id', 'Visit_date'])

# Get unique dates, aka visits
visits_per_patient = df_visits.groupby("patient_id")["Visit_date"].nunique()

# Plot
plt.figure()
plt.hist(visits_per_patient, bins=20, edgecolor='black', color='blue', alpha=0.7, label = 'visits')
plt.title(f"Distribution of visits per patient")

plt.axvline(max(visits_per_patient), linestyle='dotted', linewidth=2)
plt.text(max(visits_per_patient), plt.ylim()[1] * 0.9, f'Max: {max(visits_per_patient)}')

plt.savefig(to_path + "visits_histogram.png")


# ---- Age distribution ---- #
df_ages = timeline_df[['patient_id','Event','Value']].copy()[timeline_df['Event'] == 'Age_at_diagnosis']
df_ages["Value"] = pd.to_numeric(df_ages["Value"], errors="coerce")

patient_ages = df_ages.groupby("patient_id")["Value"].first()

# Plot
plt.figure()
plt.hist(patient_ages, bins=20, edgecolor='black', color='blue', alpha=0.7, label = 'visits')
plt.title(f"Age at diagnosis distribution")
plt.axvline(patient_ages.mean(), linestyle='dotted', linewidth=2)
plt.text(patient_ages.mean(), plt.ylim()[1] * 0.9, f'Average: {patient_ages.mean():.2f}')
plt.savefig(to_path + "ages_histogram.png")