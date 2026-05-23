#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 5/5/2026


# Import libraries
import pandas as pd

# Load file
path = "../data/"
df = pd.read_csv(path + 'filled_feature_table.csv')

# ---- Split into train/test ---- #
# Random shuffling the rows
df_shuffle = df.sample(frac=1, random_state=42).reset_index(drop=True)
# Find index position for 80 percent of entries
cut_80 = int(len(df_shuffle)*0.8)
df_train = df_shuffle[:cut_80]
df_test = df_shuffle[cut_80:]

# ---- Save ---- #
outdir = "../data/"
df_train.to_csv(outdir + "train_filled_feature_table.csv", index=False)
df_test.to_csv(outdir + "test_filled_feature_table.csv", index=False)