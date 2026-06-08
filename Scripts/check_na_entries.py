#!/usr/local/anaconda3-2024.10-1/bin/python3
# The purpose of this script is to get an idea of the number of NA entries in the feature table, 
# and to create and save a table with the counts and percentages of NA entries per column.

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Load in feature table
df = pd.read_csv(f"../data/feature_table.csv")

# Check number and percentage of NA entries per column
na_check = pd.DataFrame({
    'NA_number': df.isnull().sum(),
    '%_NA/all': (df.isnull().sum() / len(df) * 100).round(2),
    'total': len(df)
}).sort_values('NA_number', ascending=False)

# Save table as image
fig, ax = plt.subplots(figsize=(12, len(na_check)*0.4 + 1))
ax.axis('tight')
ax.axis('off')
table_ax = table(ax, na_check,
                colWidths=[0.3, 0.2, 0.2, 0.2],
                cellLoc='center',
                loc='center')
table_ax.auto_set_font_size(False)
table_ax.set_fontsize(12)
table_ax.scale(1.2, 2)
plt.title('NA Summary', fontsize=16, pad=20)
plt.savefig(f'../data/figures/table_na_check.png', dpi=300, bbox_inches='tight', facecolor='white')
