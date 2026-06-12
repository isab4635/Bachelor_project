#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 20/5/2026

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

# Settings
label = "mtx_stopped"
path = "../../data/"
outdir = "../../results/mtx_models/"

# Load train and test data
df_train = pd.read_csv(path + 'scaled_train_feature_table.csv')
df_test = pd.read_csv(path + 'scaled_test_feature_table.csv')

if label == "mtx_stopped":
    df_train.drop(columns=["biologic_added"], inplace=True)
    df_test.drop(columns=["biologic_added"], inplace=True)
elif label == "biologic_added":
    df_train.drop(columns=["mtx_stopped"], inplace=True)
    df_test.drop(columns=["mtx_stopped"], inplace=True)
else:
    raise ValueError("Label must be either 'mtx_stopped' or 'biologic_added'")

# Drop features strongly correlated with DAs
df_train.drop(columns=["CDAI", "SDAI"], inplace=True)
df_test.drop(columns=["CDAI", "SDAI"], inplace=True)

# Split into X and y
X_train = df_train.drop(columns=[label])
y_train = df_train[label]

X_test = df_test.drop(columns=[label])
y_test = df_test[label]

# Split train into train and validation
X_val = X_train.sample(frac=0.2, random_state=42)
y_val = y_train.loc[X_val.index]

X_train = X_train.drop(index=X_val.index)
y_train = y_train.drop(index=y_val.index)


# ---- Training models ---- #
logistic_model = LogisticRegression(random_state=42, C=0.1, class_weight="balanced")
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_leaf=10, max_depth=5, criterion="entropy")
random_forest_model.fit(X_train, y_train)

decision_tree_model = DecisionTreeClassifier(random_state=42, min_samples_leaf=5, max_depth=5, criterion="entropy")
decision_tree_model.fit(X_train, y_train)

gbc = GradientBoostingClassifier(random_state=42, subsample=1, learning_rate=0.05, n_estimators=100, min_samples_leaf=1, max_depth=3)
gbc.fit(X_train, y_train)

# Get predictions
y_pred_logistic = logistic_model.predict_proba(X_val)[:, 1]
y_pred_rf = random_forest_model.predict_proba(X_val)[:, 1]
y_pred_dt = decision_tree_model.predict_proba(X_val)[:, 1]
y_pred_gbc = gbc.predict_proba(X_val)[:, 1]


pred_df = pd.DataFrame(
    {'Logistic': y_pred_logistic, 'RandomForest': y_pred_rf, 'DecisionTree': y_pred_dt, 'GradientBoosting': y_pred_gbc})

# Determine optimal thresholds based on different priorities
for model in ["Logistic", "RandomForest", "DecisionTree", "GradientBoosting"]:
    precision, recall, thresholds = precision_recall_curve(y_val, pred_df[model])

    # Optimizing precision
    best_idx_precision = np.argmax(precision[:-1])
    best_threshold_precision = thresholds[best_idx_precision]

    print(f"Best threshold for {model} f_precision: {best_threshold_precision}")

    # Optimizing f_0.5
    b = 0.5
    f_05 = (1 + b**2) * precision[:-1] * recall[:-1] / (b**2 * precision[:-1] + recall[:-1] + 1e-8) # 1e-8 to avoid zero division

    best_idx_05 = np.argmax(f_05)
    best_threshold_05 = thresholds[best_idx_05]

    print(f"Best threshold for {model} f_05: {best_threshold_05}")

    # Optimizing f1-score
    b = 1
    f_1 = (1 + b**2) * precision[:-1] * recall[:-1] / (b**2 * precision[:-1] + recall[:-1] + 1e-8) # 1e-8 to avoid zero division

    best_idx_1 = np.argmax(f_1)
    best_threshold_1 = thresholds[best_idx_1]

    print(f"Best threshold for {model} f_1: {best_threshold_1}")

