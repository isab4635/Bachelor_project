#!/usr/local/anaconda3-2024.10-1/bin/python3
# Goal is to investigate the relationship between biologic_added and mtx_success, 
# and whether there are significant differences in the distribution of mtx_success between those with and without biologic_added.

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas.plotting import table
import math

outdir = "../results/bio_vs_mtx_check/"
path = "../data/"

# Open the feature table
df_features = pd.read_csv(path + 'filled_feature_table.csv')

df_mtx_stop = df_features[df_features["mtx_stopped"] == 1]
df_mtx_con = df_features[df_features["mtx_stopped"] == 0]
df_no_bio = df_features[df_features["biologic_added"] == 0]
df_bio = df_features[df_features["biologic_added"] == 1]

results_categorical = []

# Chi2 test (Method 7.20 in stat enotes)
data = df_features[["mtx_stopped", "biologic_added"]].dropna()
print(len(data))
try:
        contingency_table = pd.crosstab(data["mtx_stopped"], data["biologic_added"])
        print(contingency_table)
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Expected: {expected}")
        print(f"p-value: {p_val}")
except:
        print(f"Chi-square test failed.")
        p_val = None

# Compute odds ratio and confidence interval
OR = (contingency_table.loc[0, 0] * contingency_table.loc[1, 1]) / (contingency_table.loc[0, 1] * contingency_table.loc[1, 0])
upper_95_CI = math.exp(math.log(OR) + 1.96* math.sqrt(1/contingency_table.loc[1, 1] + 1/contingency_table.loc[1, 0] + 1/contingency_table.loc[0, 1] + 1/contingency_table.loc[0, 0]))
lower_95_CI = math.exp(math.log(OR) - 1.96* math.sqrt(1/contingency_table.loc[1, 1] + 1/contingency_table.loc[1, 0] + 1/contingency_table.loc[0, 1] + 1/contingency_table.loc[0, 0]))

# Get proportions of fail and success for each category
count_bio_added_mtx_stopped = contingency_table.loc[1, 1]
count_no_bio_mtx_stopped = contingency_table.loc[1, 0]

count_mtx_con_bio_added = contingency_table.loc[0, 1]
count_mtx_stop_bio_added = contingency_table.loc[1, 1]

# Percentages within feature == 1
pct_bio = 100 * count_bio_added_mtx_stopped / contingency_table.loc[1, :].sum()
pct_no_bio = 100 * count_no_bio_mtx_stopped / contingency_table.loc[1, :].sum()

pct_mtx_con = 100 * count_mtx_con_bio_added / contingency_table.loc[:, 1].sum()
pct_mtx_stop = 100 * count_mtx_stop_bio_added / contingency_table.loc[:, 1].sum()

# Store results
results_categorical.append({
        "Biologic added / \n MTX stopped (%)": f"{count_bio_added_mtx_stopped} ({pct_bio:.1f}%)",
        "Biologic not added / \n MTX stopped (%)": f"{count_no_bio_mtx_stopped} ({pct_no_bio:.1f}%)",
        "MTX cont. / \n Biologic added (%)": f"{count_mtx_con_bio_added} ({pct_mtx_con:.1f}%)",
        "MTX stop / \n Biologic added (%)": f"{count_mtx_stop_bio_added} ({pct_mtx_stop:.1f}%)",
        "OR (95% CI)": f"{OR:.2f} ({lower_95_CI:.2f} - {upper_95_CI:.2f})",
        "Chi2 p-value": f"{p_val:.5f}" if p_val is not None else None
    })

# Create a dataframe for the results and save to CSV
results_df_categorical = pd.DataFrame(results_categorical)
results_df_categorical.to_csv(outdir + "stats.csv", index=False)

# Create a table of the results
fig, ax = plt.subplots(figsize=(12, len(results_df_categorical)*0.4 + 1))
ax.axis('tight')
ax.axis('off')
table_ax = table(ax, results_df_categorical,
                cellLoc='center',
                loc='center')
table_ax.auto_set_font_size(False)
table_ax.set_fontsize(8)
table_ax.scale(1.2, 2)

# Save the table as an image
plt.title('Statistical differences between mtx_status and biologic_added', fontsize=16, pad=20)
plt.savefig(outdir + 'table_stat_diff_binary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()