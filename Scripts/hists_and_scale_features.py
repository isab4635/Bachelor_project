#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 21/5/2026
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Settings
outdir = "../results/features/"
path = "../data/"

# Open the feature table
df_features = pd.read_csv(path + 'train_filled_feature_table.csv')
df_test = pd.read_csv(path + 'test_filled_feature_table.csv')

# Continuous numerical columns
num_cols = [
    'Age_at_diagnosis',
    'CDAI',
    'CRP',
    'DAS_28_CRP',
    'MDHAQ',
    'SDAI',
    'Swollenjoints28',
    'Vas_doctor',
    'Vas_patient_fatigue',
    'Vas_patient_global',
    'Vas_patient_pain',
    'mtx_start_year'
]

# Subset of features to plot
subset_cols = ['Age_at_diagnosis', 'CDAI', 'CRP', 'DAS_28_CRP', 'Vas_patient_global', 'mtx_start_year']

# Pretty labels for plotting
pretty_labels = {
    'Age_at_diagnosis': 'Age at diagnosis',
    'CDAI': 'CDAI',
    'CRP': 'CRP',
    'DAS_28_CRP': 'DAS28-CRP',
    'MDHAQ': 'MDHAQ',
    'SDAI': 'SDAI',
    'Swollenjoints28': 'Swollen joints',
    'Vas_doctor': 'Doctor VAS',
    'Vas_patient_fatigue': 'Fatigue VAS',
    'Vas_patient_global': 'Global VAS',
    'Vas_patient_pain': 'Pain VAS',
    'mtx_start_year': 'MTX start year'
}

# ---- Plot the distribution of the original features ---- #
# Histogram plots of the original features
fig, axes = plt.subplots(3, 4, figsize=(16, 10))

for ax, col in zip(axes.flatten(), num_cols[:11]):
    sb.histplot(data=df_features, x=col, fill=True, ax=ax)
    ax.set_title(pretty_labels.get(col, col))

sb.histplot(data=df_features, bins=9, x='mtx_start_year', fill=True, ax=axes.flatten()[11])
axes.flatten()[11].set_title(pretty_labels.get('mtx_start_year', 'mtx_start_year'))

plt.suptitle("Histograms of original numerical features")
plt.tight_layout()
plt.savefig(outdir + "before_norm_histplots.png", dpi=300, bbox_inches='tight')
plt.close()

# Histogram plots for subset of features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, col in zip(axes.flatten(), subset_cols[:5]):
    sb.histplot(data=df_features, x=col, fill=True, ax=ax)
    ax.set_title(pretty_labels.get(col, col))

sb.histplot(data=df_features, bins=9, x='mtx_start_year', fill=True, ax=axes.flatten()[5])
axes.flatten()[5].set_title(pretty_labels.get('mtx_start_year', 'mtx_start_year'))

plt.suptitle("Histograms of original numerical features")
plt.tight_layout()
plt.savefig(outdir + "before_norm_hist_subset.png", dpi=300, bbox_inches='tight')
plt.close()


# ---- Scale and normalize the numerical features ---- #
# Skewed columns to log-transform
#skewed = ['CRP', 'Swollenjoints28', 'CDAI', 'SDAI', 'MDHAQ', 'Vas_doctor']
skewed = ['CRP']
df_features[skewed] = df_features[skewed].apply(lambda x: np.log1p(x))
df_test[skewed] = df_test[skewed].apply(lambda x: np.log1p(x))

# Standardize the numerical features
df_test[num_cols] = (df_test[num_cols] - df_features[num_cols].mean()) / df_features[num_cols].std()
df_features[num_cols] = (df_features[num_cols] - df_features[num_cols].mean()) / df_features[num_cols].std()

scaled_df = df_features[num_cols]


# ---- Plot the distribution of the scaled features ---- #
# Plot the scaled features
plt.figure(figsize=(14, 6))
sb.boxplot(data=scaled_df)
plt.xticks(ticks=range(len(num_cols)), labels=[pretty_labels.get(col, col) for col in num_cols],
           rotation=45,
           ha='right')
plt.title("Boxplots of scaled numerical features")
plt.tight_layout()
plt.savefig(outdir + "after_norm_boxplots.png", dpi=300, bbox_inches='tight')
plt.close()

# KDE plots of the scaled features in one plot
plt.figure(figsize=(14, 6))
sb.kdeplot(data=scaled_df, palette=sb.color_palette("husl", len(num_cols)))
plt.legend(labels=[pretty_labels.get(col, col) for col in num_cols])
plt.title("KDE plots of scaled numerical features")
plt.tight_layout()
plt.savefig(outdir + "after_norm_kdeplot.png", dpi=300, bbox_inches='tight')
plt.close()

# KDE plots of the scaled features in subplots
fig, axes = plt.subplots(3, 4, figsize=(10, 7))

for ax, col in zip(axes.flatten(), num_cols):
    sb.kdeplot(data=scaled_df, x=col, fill=True, ax=ax)
    ax.set_title(pretty_labels.get(col, col))

plt.suptitle("KDE plots of scaled numerical features")
plt.tight_layout()
plt.savefig(outdir + "after_norm_kdeplots.png", dpi=300, bbox_inches='tight')
plt.close()

# Histogram plots for subset of features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, col in zip(axes.flatten(), subset_cols[:5]):
    sb.histplot(data=scaled_df, x=col, fill=True, ax=ax)
    ax.set_title(pretty_labels.get(col, col))

sb.histplot(data=scaled_df, bins=9, x='mtx_start_year', fill=True, ax=axes.flatten()[5])
axes.flatten()[5].set_title(pretty_labels.get('mtx_start_year', 'mtx_start_year'))

plt.suptitle("Histograms of scaled numerical features")
plt.tight_layout()
plt.savefig(outdir + "after_norm_hist_subset.png", dpi=300, bbox_inches='tight')
plt.close()


# ---- Save the scaled features ---- #
# Save the scaled features to a new CSV file
df_features.to_csv(path + "scaled_train_feature_table.csv", index=False)
df_test.to_csv(path + "scaled_test_feature_table.csv", index=False)
