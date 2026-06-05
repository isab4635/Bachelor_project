#!/usr/local/anaconda3-2024.10-1/bin/python3
# Script with statistical analysis of the features for the biologic subquestion. It computes summary statistics, performs statistical tests, calculates AUC-ROC for each feature, and creates visualizations such as violin plots and correlation heatmaps.

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas.plotting import table
import seaborn as sb
from sklearn.metrics import roc_auc_score
import math
import numpy as np

# Settings
label="biologic_added"
outdir = "../../results/biologic_features/"
path = "../../data/"

# Open the feature table
df_features = pd.read_csv(path + 'train_filled_feature_table.csv')

if label == "mtx_stopped":
    df_features.drop(columns=["biologic_added"], inplace=True)
elif label == "biologic_added":
    df_features.drop(columns=["mtx_stopped"], inplace=True)
else:
    raise ValueError("Label must be either 'mtx_stopped' or 'biologic_added'")

df_fail = df_features[df_features[label] == 0]
df_success = df_features[df_features[label] == 1]

categorical_cols = ["Koen", "IgM_RF", "Anti_CCP"]
numerical_cols = df_features.columns.drop([label,"Koen" , "IgM_RF", "Anti_CCP"])
all_cols = df_features.columns.drop(label)


# ---- Stats for numerical features ---- #
results_numerical = []

for col in numerical_cols:
    # Make subsets
    subset_fail = df_fail[col].dropna()
    subset_success = df_success[col].dropna()

    # Summary stats
    median_fail = subset_fail.median()
    median_success = subset_success.median()

    # Statistical test (Mann–Whitney)
    try:
        stat, p_value = stats.mannwhitneyu(subset_fail, subset_success, alternative='two-sided')
    except:
        p_value = None

    # Compute AUC
    try:
        data = df_features[[col, label]].dropna()
        auc = roc_auc_score(data[label], data[col])
    except:
        auc = None

    # AUC difference from 0.5
    if auc is not None:
        auc_diff = abs(auc - 0.5)
    else:
        auc_diff = None

    # Store results
    results_numerical.append({
        "Feature": col,
        "median_not [IQR]": f"{median_fail:.2f} [{subset_fail.quantile(0.25):.2f} - {subset_fail.quantile(0.75):.2f}]",
        "median_added [IQR]": f"{median_success:.2f} [{subset_success.quantile(0.25):.2f} - {subset_success.quantile(0.75):.2f}]",
        "MWU p-value": round(p_value, 5) if p_value is not None else None,
        "auc-roc": round(auc, 5) if auc is not None else None,
        "auc_diff_from_0.5": round(auc_diff, 5) if auc_diff is not None else None
    })

    # Violin plot
    plt.figure(figsize=(6, 4))
    sb.violinplot(x=label, y=col, data=df_features.dropna(subset=[col]))
    plt.title(f"Violin plot of {col} vs {label}")
    plt.tight_layout()
    plt.savefig(outdir + f"{col}_violinplot.png")
    plt.close()

# Create a dataframe for the results and save to CSV
results_df_numerical = pd.DataFrame(results_numerical)
results_df_numerical.sort_values(by=["auc_diff_from_0.5"], inplace=True, ascending=False)
results_df_numerical.to_csv(outdir + "feature_stats_numerical.csv", index=False)

# Create a table of the results
fig, ax = plt.subplots(figsize=(12, len(results_df_numerical)*0.4 + 1))
ax.axis('tight')
ax.axis('off')
table_ax = table(ax, results_df_numerical,
                cellLoc='center',
                loc='center')
table_ax.auto_set_font_size(False)
table_ax.set_fontsize(12)
table_ax.scale(1.2, 2)

plt.title('Statistical differences of numerical features', fontsize=16, pad=20)
plt.savefig(outdir + 'table_stat_diff_numerical.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- Stats for categorical features ---- #
results_categorical = []

for col in categorical_cols:
    # Chi2 test (Method 7.20 in stat enotes)
    data = df_features[[col, label]].dropna()
    try:
        contingency_table = pd.crosstab(data[label], data[col])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    except:
        print(f"Chi-square test failed for {col}. Skipping.")
        p_val = None

    # Compute AUC
    try:
        auc = roc_auc_score(data[label], data[col])
    except:
        print(f"Auc failed for {col}. Skipping.")
        auc = None

    # AUC difference from 0.5
    if auc is not None:
        auc_diff = abs(auc - 0.5)
    else:
        auc_diff = None

    # Compute odds ratio and confidence interval
    OR = (contingency_table.loc[0, 0] * contingency_table.loc[1, 1]) / (contingency_table.loc[0, 1] * contingency_table.loc[1, 0])
    upper_95_CI = math.exp(math.log(OR) + 1.96* math.sqrt(1/contingency_table.loc[1, 1] + 1/contingency_table.loc[1, 0] + 1/contingency_table.loc[0, 1] + 1/contingency_table.loc[0, 0]))
    lower_95_CI = math.exp(math.log(OR) - 1.96* math.sqrt(1/contingency_table.loc[1, 1] + 1/contingency_table.loc[1, 0] + 1/contingency_table.loc[0, 1] + 1/contingency_table.loc[0, 0]))

    # Get proportions of fail and success for each category
    # Count feature == 1
    count_fail = contingency_table.loc[0, 1]
    count_success = contingency_table.loc[1, 1]

    # Percentages within feature == 1
    pct_fail = 100 * count_fail / contingency_table.loc[0].sum()
    pct_success = 100 * count_success /contingency_table.loc[1].sum()

    # Store results
    results_categorical.append({
        "Feature": col,
        "Not_added (%)": f"{count_fail} ({pct_fail:.1f}%)",
        "Added (%)": f"{count_success} ({pct_success:.1f}%)",
        "OR (95% CI)": f"{OR:.2f} ({lower_95_CI:.2f} - {upper_95_CI:.2f})",
        "Chi2 p-value": round(p_val,5) if p_val is not None else None,
        "auc-roc": round(auc, 5) if auc is not None else None,
        "auc_diff_from_0.5": round(auc_diff, 5) if auc_diff is not None else None
    })

# Create a dataframe for the results and sort it by AUC difference from 0.5
results_df_categorical = pd.DataFrame(results_categorical)
results_df_categorical.sort_values(by=["auc_diff_from_0.5"], inplace=True, ascending=False)

results_df_categorical.to_csv(outdir + "feature_stats_categorical.csv", index=False)

# Create a table of the results
fig, ax = plt.subplots(figsize=(12, len(results_df_categorical)*0.4 + 1))
ax.axis('tight')
ax.axis('off')
table_ax = table(ax, results_df_categorical,
                cellLoc='center',
                loc='center')
table_ax.auto_set_font_size(False)
table_ax.set_fontsize(12)
table_ax.scale(1.2, 2)

plt.title('Statistical differences of binary features', fontsize=16, pad=20)
plt.savefig(outdir + 'table_stat_diff_binary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- Correlation heatmaps ---- #
corr = df_features[numerical_cols].corr(method="pearson")
sb.heatmap(corr, annot=True, annot_kws={'size': 8}, cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Pearson)")
plt.tight_layout()
plt.savefig(outdir + "corr_matrix_pearson.png", dpi=300, bbox_inches='tight')
plt.close()

corr = df_features[numerical_cols].corr(method="spearman")
sb.heatmap(corr, annot=True, annot_kws={'size': 8}, cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Spearman)")
plt.tight_layout()
plt.savefig(outdir + "corr_matrix_spearman.png", dpi=300, bbox_inches='tight')
plt.close()

#chierarchal clustering

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

data = df_features[numerical_cols]
plt.figure(figsize=(12,5))
dissimilarity = 1 - abs(corr)

Z = linkage(squareform(dissimilarity), 'complete')

dendrogram(Z, labels=data.columns, orientation='top',
           leaf_rotation=90);
plt.title("Dendrogram of features")
plt.tight_layout()
plt.savefig(outdir + "corr_dendrogram.png", dpi=300, bbox_inches='tight')
plt.close()

threshold = 0.8
labels = fcluster(Z, threshold, criterion='distance')
labels_order = np.argsort(labels)

for idx, i in enumerate(data.columns[labels_order]):
    if idx == 0:
        clustered = pd.DataFrame(data[i])
    else:
        df_to_append = pd.DataFrame(data[i])
        clustered = pd.concat([clustered, df_to_append], axis=1)
plt.figure(figsize=(15,10))
correlations = clustered.corr()
sb.heatmap(round(correlations,2), cmap='coolwarm', annot=True,
            annot_kws={"size": 12}, vmin=-1, vmax=1);
plt.title("Correlation heatmap - hierarchical clustering")
plt.tight_layout()
plt.savefig(outdir + "corr_hierarchical_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# ---- Correlation heatmaps on patients---- #
df = df_features[numerical_cols].T
corr = df.corr(method="spearman")
print(corr)

dissimilarity = (1 - corr)/2
Z = linkage(squareform(dissimilarity), 'complete')

print(np.allclose(dissimilarity, dissimilarity.T))
print(np.diag(dissimilarity))

threshold = 0.9
labels = fcluster(Z, threshold, criterion='distance')
labels_order = np.argsort(labels)

ordered_corr = corr.iloc[labels_order, labels_order]

plt.figure(figsize=(15,10))
sb.heatmap(round(ordered_corr,2), cmap='coolwarm', annot=False,
            vmin=0, vmax=1)
plt.title("Correlation heatmap - hierarchical clustering")
plt.tight_layout()
plt.savefig(outdir + "corr_hierarchical_heatmap_patients.png", dpi=300, bbox_inches='tight')
plt.close()