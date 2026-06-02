#!/usr/local/anaconda3-2024.10-1/bin/python3
# The principle is to analyze the prescriptions per patients, 
# and to construct new csv, and lists of patients that within a year have been prescribed biologic or ts_dmard or both

#Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('../data/final_timeline_csv_after_all_exclusions.csv')
X = df.copy()

# Focus on prescriptions within 12 months after starting MTX (+ 3 months to account for breaks in prescriptions)
X_drugs = X[X['Event'].isin(['Prescription_start', 'Prescription_stop']) & X['months_since_mtx'] <= 15]

# Initialize counts and group drugs
y = 0
x = 0
set_patients_with_prescriptions_same_day = set()
fig, ax = plt.subplots(constrained_layout=True)
treatments = {'BIOLOGIC_BIMEKIZUMAB': 'biologic_bimekizymab', 'BIOLOGIC_TOCILIZUMAB_SC': 'biologic_tocilizumab', 'BIOLOGIC_ERELZI': 'biologic_etanercept', 'BIOLOGIC_UPADACITINIB': 'biologic_upadacitinib',
            'BIOLOGIC_AMGEVITA': 'biologic_adalimumab', 'BIOLOGIC_RUXIENCE': 'biologic_rituximab', 'BIOLOGIC_BARICITINIB': 'biologic_baricitinib', 'BIOLOGIC_USTEKINUMAB': 'biologic_ustekinumab',
            'BIOLOGIC_IMRALDI': 'biologic_adalimumab', 'BIOLOGIC_CERTOLIZUMAB': 'biologic_certolizumab', 'BIOLOGIC_GOLIMUMAB': 'biologic_golimumab', 'BIOLOGIC_SARILUMAB': 'biologic_sarilumab',
            'BIOLOGIC_REMSIMA': 'biologic_infliximab', 'BIOLOGIC_ABATACEPT': 'biologic_abatacept', 'BIOLOGIC_PROJECT_MEDICINE': 'biologic_biologic_project_medicine',
            'BIOLOGIC_TOFACITINIB': 'biologic_tofacitinib', 'BIOLOGIC_SECUKINUMAB': 'biologic_secukinumab', 'BIOLOGIC_MYCOPHENOLAT': 'biologic_mycophenolatmofetil', 'BIOLOGIC_BELIMUMAB': 'biologic_belimumab',
            'BIOLOGIC_ZESSLY': 'biologic_infliximab', 'BIOLOGIC_BENEPALI': 'biologic_etanercept', 'BIOLOGIC_FLIXABI': 'biologic_infliximab', 'BIOLOGIC_TOCILIZUMAB': 'biologic_tocilizumab',
            'BIOLOGIC_ADALIMUMAB': 'biologic_adalimumab', 'BIOLOGIC_FILGOTINIB': 'biologic_filgotinib', 'BIOLOGIC_INFLECTRA': 'biologic_infliximab', 'BIOLOGIC_ETANERCEPT': 'biologic_etanercept',
            'BIOLOGIC_IXEKINUMAB': 'biologic_ixekizumab', 'BIOLOGIC_ABATACEPT_SC': 'biologic_abatacept', 'BIOLOGIC_UZPRUVO': 'biologic_ustekinumab', 'BIOLOGIC_OTEZLA': 'biologic_apremilast',
            'BIOLOGIC_RIXATHON': 'biologic_rituximab', 'BIOLOGIC_INFLIXIMAB': 'biologic_infliximab', 'BIOLOGIC_ANAKINRA': 'biologic_anakinra', 'BIOLOGIC_RITUXIMAB': 'biologic_rituximab',
            'BIOLOGIC_HYRIMOZ': 'biologic_adalimumab', 'DMARD_OTHER': 'DMARD_OTHER', 'DMARD_CYCLO_IV': 'dmard_cyclo', 'DMARD_SZS': 'dmard_szs', 'DMARD_MTX_SC': 'dmard_mtx', 'DMARD_CYCLO': 'dmard_cyclo',
            'DMARD_GOLD': 'dmard_gold', 'DMARD_MTX': 'dmard_mtx', 'DMARD_CYA': 'dmard_cya', 'DMARD_FOLIMET': 'dmard_folimet', 'DMARD_ISOVORIN': 'dmard_isovarin','DMARD_MTX_IM': 'dmard_mtx',
            'DMARD_AZATH': 'dmard_azath', 'DMARD_PENIC': 'dmard_penic', 'DMARD_LEFLUN': 'dmard_leflun', 'DMARD_CHLOROCHINE': 'dmard_chlorochine', 'NSAID_ACEMETASINE': 'nsaid_acemetasine',
            'NSAID_DICLOFENAC': 'nsaid_diclofenac', 'NSAID_MELOXICAM': 'nsaid_meloxicam', 'NSAID_IBUPROFEN': 'nsaid_ibuprofen', 'NSAID_OTHER': 'NSAID_OTHER', 'NSAID_NAPROXEN': 'nsaid_naproxen',
            'NODRUG': 'NODRUG', 'NODRUG_NSAID_NO_INDICATION': 'NODRUG_NSAID_NO_INDICATION', 'MEDICINE_COLCHICIN': 'medicine_colchicin', 'PREDNISOLON': 'prednisolon',
            'MEDICINE_FEBUXOSTAT': 'medicine_febuxostat', 'MEDICINE_HYDROCORTISON': 'medicine_hydrocortison', 'MEDICINE_ALLOPURINOL': 'medicine_allopurinol', 'NON_BIOLOGIC': 'NON_BIOLOGIC'}
biologic = ["biologic_bimekizymab", "biologic_tocilizumab", "biologic_etanercept", "biologic_adalimumab", "biologic_rituximab", "biologic_ustekinumab", "biologic_infliximab"]
cs_dmard_excl_mtx = ["DMARD_OTHER", "dmard_cyclo", "dmard_szs", "dmard_gold", "dmard_cya", "dmard_folimet", "dmard_isovarin","dmard_azath", "dmard_penic", "dmard_leflun", "dmard_chlorochine"]
others = ["nsaid_acemetasine", "nsaid_diclofenac", "nsaid_meloxicam", "nsaid_ibuprofen", "NSAID_OTHER", "nsaid_naproxen", 'NODRUG', 'NODRUG_NSAID_NO_INDICATION', 'medicine_colchicin', 'prednisolon',
         'medicine_febuxostat', 'medicine_hydrocortison', 'medicine_allopurinol', 'NON_BIOLOGIC']
mtx = ["dmard_mtx"]
ts_dmard = ["biologic_baricitinib", "biologic_upadacitinib"]

X_drugs['Value'] = X_drugs['Value'].replace(treatments)

# Prepare for histogram
cs_dmard_count_dic = {}
for entry in cs_dmard_excl_mtx:
     cs_dmard_count_dic[entry] = 0

ts_dmard_count_dic = {}
for entry in ts_dmard:
     ts_dmard_count_dic[entry] = 0

b_dmard_count_dic = {}
for entry in biologic:
     b_dmard_count_dic[entry] = 0

treatments_count_dic = {}
treatments_count_dic["Biologic"] = 0
treatments_count_dic["Biologic_and_ts_DMARD"] = 0
treatments_count_dic["ts_DMARD"] = 0
treatments_count_dic["no_Biologic_nor_ts_DMARD"] = 0

subset_drugs_together = pd.DataFrame()
patients_on_biologic_and_ts = []
patients_on_biologic = []
patients_on_ts = []

# Loop through each patient and analyze their prescriptions
for patient in X['patient_id'].unique():

#creating subsets
        subset_drugs = X_drugs[X_drugs['patient_id'] == patient].copy()
        subset_prescription_start = subset_drugs[subset_drugs['Event'] == 'Prescription_start'].copy()
        subset_prescription_stop = subset_drugs[subset_drugs['Event'] == 'Prescription_stop'].copy()

#checking if prescriptions missing
        if subset_drugs.empty:
                print(f"Missing prescrption start date values for patient {patient}.")
                x+=1
                continue
        print(f'patient:{patient}\n subset before excluding breaks: {subset_drugs}\n')

#Excluding breaks in prescription smaller than 3 months
        for drug in subset_prescription_start["Value"].unique():
                prescription_start = subset_prescription_start[subset_prescription_start["Value"] == drug].copy()
                prescription_stop = subset_prescription_stop[subset_prescription_stop["Value"] == drug].copy()
                if not prescription_stop.empty:
                        starts = pd.to_datetime(prescription_start['Date'])
                        stops = pd.to_datetime(prescription_stop['Date'])
                        differences = (stops.values[:, None] - starts.values)
                        window = ((differences <= pd.Timedelta(0)) &
                                          (differences >= -pd.Timedelta(days=90)))
                        too_close_stop = window.any(axis=1)
                        prescription_stop = prescription_stop.loc[~too_close_stop]
                        subset_prescription_stop = pd.concat([
                                         subset_prescription_stop[subset_prescription_stop["Value"] != drug],
                                         prescription_stop], ignore_index=True)
        subset_drugs = subset_drugs[~subset_drugs["Event"].isin(["Prescription_stop", "Prescription_start"])]
        subset_drugs = pd.concat([subset_drugs, subset_prescription_start, subset_prescription_stop], ignore_index=True)
        subset_drugs.sort_values('Date', inplace = True)
        subset_drugs.reset_index(drop=True, inplace = True)
        print(f'subset after excluding breaks: {subset_drugs}\n')
#only the first year
        subset_drugs = subset_drugs[subset_drugs['months_since_mtx'] <= 12].copy()

#counting treatments and grouping them
        starts = subset_drugs[(subset_drugs['Event'] == 'Prescription_start') &
                              (subset_drugs['Value'].isin(biologic))]['Value'].value_counts()

        for drug, count in starts.items():
              b_dmard_count_dic[drug] += count

        starts = subset_drugs[(subset_drugs['Event'] == 'Prescription_start') &
                              (subset_drugs['Value'].isin(ts_dmard))]['Value'].value_counts()

        for drug, count in starts.items():
              ts_dmard_count_dic[drug] += count

        starts = subset_drugs[(subset_drugs['Event'] == 'Prescription_start') &
                              (subset_drugs['Value'].isin(cs_dmard_excl_mtx))]['Value'].value_counts()

        for drug, count in starts.items():
              cs_dmard_count_dic[drug] += count

#new columns to account for whether patient is on biologic, ts_dmard, cs_dmard_excl_mtx or mtx at a given time point
        start = (subset_drugs['Event'] == 'Prescription_start') & subset_drugs['Value'].isin(biologic)
        stop  = (subset_drugs['Event'] == 'Prescription_stop')  & subset_drugs['Value'].isin(biologic)
        diff = np.zeros(len(subset_drugs))
        diff[start] = 1
        diff[stop]  = -1
        subset_drugs['Biologic'] = np.clip(diff.cumsum(), 0, 1)

        start = (subset_drugs['Event'] == 'Prescription_start') & subset_drugs['Value'].isin(ts_dmard)
        stop  = (subset_drugs['Event'] == 'Prescription_stop')  & subset_drugs['Value'].isin(ts_dmard)
        diff = np.zeros(len(subset_drugs))
        diff[start] = 1
        diff[stop]  = -1
        subset_drugs['ts_DMARD'] = np.clip(diff.cumsum(), 0, 1)

        start = (subset_drugs['Event'] == 'Prescription_start') & subset_drugs['Value'].isin(cs_dmard_excl_mtx)
        stop  = (subset_drugs['Event'] == 'Prescription_stop')  & subset_drugs['Value'].isin(cs_dmard_excl_mtx)
        diff = np.zeros(len(subset_drugs))
        diff[start] = 1
        diff[stop]  = -1
        subset_drugs['cs_DMARD_excl_MTX'] = np.clip(diff.cumsum(), 0, 1)

        start = (subset_drugs['Event'] == 'Prescription_start') & subset_drugs['Value'].isin(mtx)
        stop  = (subset_drugs['Event'] == 'Prescription_stop')  & subset_drugs['Value'].isin(mtx)
        diff = np.zeros(len(subset_drugs))
        diff[start] = 1
        diff[stop]  = -1
        subset_drugs['MTX'] = np.clip(diff.cumsum(), 0, 1)

        subset_drugs['Sum_bio_and_ts'] = subset_drugs['Biologic'] + subset_drugs['ts_DMARD']

        subset_drugs_together = pd.concat([subset_drugs_together, subset_drugs], ignore_index=True)

#adding the particular patient to treatment group by prescriptions
        if (subset_drugs["Sum_bio_and_ts"] == 2).any():
                treatments_count_dic["Biologic_and_ts_DMARD"] += 1
                patients_on_biologic_and_ts.append(patient)
        elif (subset_drugs["Biologic"] == 1).any():
                treatments_count_dic["Biologic"] += 1
                patients_on_biologic.append(patient)
        elif (subset_drugs["ts_DMARD"] == 1).any():
                treatments_count_dic["ts_DMARD"] += 1
                patients_on_ts.append(patient)
        else:
                treatments_count_dic["no_Biologic_nor_ts_DMARD"] += 1

# Print the results
print(f'Total number of patients: {subset_drugs_together['patient_id'].nunique()}')
print(f"Number of patients missing prescriptions: {x}")
print(f"Number of prescription start and stop on same day: {y}")
print(f'Number of patients with prescription start stop same day: {len(set_patients_with_prescriptions_same_day)}')
print(treatments_count_dic)

# Plot results
plt.figure()
pd.Series(treatments_count_dic).plot.bar()
plt.title('Final treatment count')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.savefig(f'../data/figures/final_treatment_count.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
pd.Series(b_dmard_count_dic).plot.bar()
plt.title('Biologic Drug Usage Histogram')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.savefig(f'../data/figures/biologic_drug_usage_hist.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
pd.Series(ts_dmard_count_dic).plot.bar()
plt.title('ts_DMARD Drug Usage Histogram')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.savefig(f'../data/figures/ts_DMARD_drug_usage_hist.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
pd.Series(cs_dmard_count_dic).plot.bar()
plt.title('cs_DMARD_no_MTX Drug Usage Histogram')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.savefig(f'../data/figures/cs_dmard_drug_usage_hist.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the subset with drug information and the lists of patients in each treatment group
subset_drugs_together.to_csv(f'../data/subset_prescriptions.csv', index=False)

with open(f'../data/list_patients_on_bio_and_ts.txt', 'w') as f:
    for line in patients_on_biologic_and_ts:
        f.write(f"{line}\n")

with open(f'../data/list_patients_on_bio.txt', 'w') as f:
    for line in patients_on_biologic:
        f.write(f"{line}\n")

with open(f'../data/list_patients_on_ts.txt', 'w') as f:
    for line in patients_on_ts:
        f.write(f"{line}\n")