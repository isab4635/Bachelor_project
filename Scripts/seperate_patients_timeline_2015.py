#!/usr/local/anaconda3-2024.10-1/bin/python3
import pandas as pd
import numpy as np

df = pd.read_csv('../data/timeline_style.csv')
X = df.copy()

patients_before_2015 = set()
patients_after_2015 = set()
excluded_no_diagnosis_date = []
count_no_con = 0

for patient in X['patient_id'].unique():

#creating subsets to adjust them
        subset = X[X['patient_id'] == patient]
        Diagnosis = subset[subset['Event'] == 'Diagnosis_date']

#checking if diagnosis date missing
        if Diagnosis.empty:
                print(f"Missing diagnosis date for patient {patient}, tryin to find constructed date.")
                print("............................................")
                Diagnosis = subset[subset['Event'] == 'Diagnosis_date_con']
                if Diagnosis.empty:
                     print(f"No constructed date found - patient won't be included in the model")
                     count_no_con += 1
                     excluded_no_diagnosis_date.append(patient)
                     continue

#taking the earliest diagnosis date
        Diagnosis_date = pd.to_datetime(Diagnosis.iloc[0,1], format = "mixed")
        if Diagnosis_date < pd.to_datetime('2015-01-01'):
                patients_before_2015.add(patient)
        else:
                patients_after_2015.add(patient)

df_before_2015 = X[X['patient_id'].isin(patients_before_2015)]
df_after_2015 = X[X['patient_id'].isin(patients_after_2015)]

print(f'number of missing constructed date: {count_no_con}')

#save seperated patients dataset
df_before_2015.to_csv(f'../data/timeline_diagnosis_before_2015.csv', index=False)
df_after_2015.to_csv(f'../data/timeline_diagnosis_after_2015.csv', index=False)
with open(f'../data/excluded/timeline_no_diagnosis_date.txt', 'w') as f:
        for line in excluded_no_diagnosis_date:
        f.write(f"{line}\n")



#OLD VERSION:
#!/usr/local/anaconda3-2024.10-1/bin/python3
import pandas as pd
import numpy as np

df = pd.read_csv('../data/timeline_style.csv')
X = df.copy()

patients_before_2015 = set()
patients_after_2015 = set()

for patient in X['patient_id'].unique():

#creating subsets to adjust them
        subset = X[X['patient_id'] == patient]
        Diagnosis = subset[subset['Event'] == 'Diagnosis_date']

#checking if diagnosis date missing
        if Diagnosis.empty:
                print(f"Missing diagnosis date for patient {patient}.")
                continue

#checking if more than one diagnosis date
        if len(Diagnosis) > 1:
                print(f"Warning: Multiple diagnosis dates found for patient {patient}. Using the first one.")
        #       MTX_start = pd.to_datetime(MTX_start, format = "%Y-%m-%d")
        #       MTX_start = MTX_start.sort_values(by='date', ascending=True)
        #       print(MTX_start)
                Diagnosis = Diagnosis.iloc[[0]]
        Diagnosis_date = pd.to_datetime(Diagnosis.iloc[0,1], format = "%Y-%m-%d")
        if Diagnosis_date < pd.to_datetime('2015-01-01'):
                patients_before_2015.add(patient)
        else:
                patients_after_2015.add(patient)

df_before_2015 = X[X['patient_id'].isin(patients_before_2015)]
df_after_2015 = X[X['patient_id'].isin(patients_after_2015)]

#save seperated patients dataset
df_before_2015.to_csv(f'../data/timeline_diagnosis_before_2015.csv', index=False)
df_after_2015.to_csv(f'../data/timeline_diagnosis_after_2015.csv', index=False)