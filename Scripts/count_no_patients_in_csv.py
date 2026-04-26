#!/usr/local/anaconda3-2024.10-1/bin/python3
import sys
import pandas as pd
import numpy as np

dataset = sys.argv[1]

df = pd.read_csv(f'../data/{dataset}.csv', encoding = 'unicode_escape', dtype= str)
count = df["patient_id"].nunique()
print("number of unique patients in the file:")
print(count)