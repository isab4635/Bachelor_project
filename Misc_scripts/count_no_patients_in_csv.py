#!/usr/local/anaconda3-2024.10-1/bin/python3
# This script counts the number of unique patients in a called out CSV file
import sys
import pandas as pd

dataset = sys.argv[1]

df = pd.read_csv(f'../data/{dataset}.csv', encoding = 'unicode_escape', dtype= str)
count = df["patient_id"].nunique()
print("number of unique patients in the file:")
print(count)