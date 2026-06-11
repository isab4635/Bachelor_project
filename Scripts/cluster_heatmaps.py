#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 21/5/2026

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# Settings
outdir = "../results/features/"
path = "../data/"

# Open the feature table
df_features = pd.read_csv(path + 'scaled_train_feature_table.csv')

num_cols = df_features.columns.drop(["mtx_stopped", "biologic_added","Koen" , "IgM_RF", "Anti_CCP"])


# ---- Correlation and clustering of features ---- #
data = df_features[num_cols]
row_colors = pd.DataFrame({"MTX Stopped": df_features["mtx_stopped"].map({0: "#cbc3e3", 1: "indigo"}),
    "Biologic Added": df_features["biologic_added"].map({0: "#CCCCCC", 1: "black"})
})
sb.clustermap(data, method="ward", metric="euclidean", cmap="coolwarm", row_colors=row_colors,  yticklabels=False)
plt.suptitle("Clustering of patient features", y=1.02)
plt.savefig(outdir + "clustermap.png")


# ---- Clustered correlation heatmap ---- #
method = "spearman"
corr = df_features[num_cols].corr(method=method)
sb.clustermap(corr, method="complete", cmap='coolwarm', annot=True, dendrogram_ratio=(0, .2), cbar_pos=(-0.05, .3, .02, .4), figsize=(8, 7))
plt.suptitle(f"Correlation Matrix ({method})", y=1.02)
plt.savefig(outdir + f"corr_matrix_{method}.png", dpi=300, bbox_inches='tight')
plt.close()