#!/usr/local/anaconda3-2024.10-1/bin/python3
# Tuning thresholds for different models to optimize precision, recall, f1-score, and f2-score for the biologic_added label.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from math import log, log1p

# Settings
label = "biologic_added"
path = "../../data/"
outdir = "../../results/bio_models/"

# Load train and test data
df_train = pd.read_csv(path + 'train_filled_feature_table.csv')
df_test = pd.read_csv(path + 'test_filled_feature_table.csv')

df_train.drop(columns=["mtx_stopped"], inplace=True)
df_test.drop(columns=["mtx_stopped"], inplace=True)

X_train = df_train.drop(columns=["biologic_added"])
y_train = df_train["biologic_added"]

X_test = df_test.drop(columns=["biologic_added"])
y_test = df_test["biologic_added"]

#Improvements to the baseline model

#redundant featurs
X_train.drop(columns = ["SDAI", "CDAI"], inplace=True)
X_test.drop(columns = ["SDAI", "CDAI"], inplace=True)

# Split train into train and validation
X_val = X_train.sample(frac=0.2, random_state=42)
y_val = y_train.loc[X_val.index]

X_train = X_train.drop(index=X_val.index)
y_train = y_train.drop(index=y_val.index)

#handling class imbalance
#https://www.geeksforgeeks.org/machine-learning/how-does-the-classweight-parameter-in-scikit-learn-work/

random_forest_model = RandomForestClassifier(random_state=42, n_estimators=300, min_samples_leaf=10, max_depth=None, criterion="entropy")
random_forest_model.fit(X_train, y_train)

decision_tree_model = DecisionTreeClassifier(random_state=42, class_weight = "balanced", min_samples_leaf=5, max_depth=5, criterion="entropy")
decision_tree_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict_proba(X_val)[:, 1]
y_pred_dt = decision_tree_model.predict_proba(X_val)[:, 1]

# log-transforming skewed features
skewed = ['CRP', 'Swollenjoints28', 'MDHAQ']
X_train[skewed] = X_train[skewed].apply(lambda x: np.log1p(x))
X_test[skewed] = X_test[skewed].apply(lambda x: np.log1p(x))
X_val[skewed] = X_val[skewed].apply(lambda x: np.log1p(x))

#normalizing and scaling data (standardizing)
col_to_change = [col for col in X_train.columns if col not in ["IgM_RF", "Anti_CCP", "Sex"]]

X_mean = X_train[col_to_change].mean()
X_std = X_train[col_to_change].std()
X_train[col_to_change] = (X_train[col_to_change] - X_mean) / X_std
X_test[col_to_change] = (X_test[col_to_change] - X_mean) / X_std
X_val[col_to_change] = (X_val[col_to_change] - X_mean) / X_std

# Training models (logistic and RF and simple decision tree, adding boosting)
logistic_model = LogisticRegression(random_state=42, class_weight = "balanced", C=0.01)
logistic_model.fit(X_train, y_train)

gbc = GradientBoostingClassifier(random_state=42, subsample=0.8, n_estimators=200, min_samples_leaf=10, max_depth=5, learning_rate=0.01)
gbc.fit(X_train, y_train)

# Get predictions
y_pred_logistic = logistic_model.predict_proba(X_val)[:, 1]
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

    # Optimizing recall
    best_idx_recall = np.argmax(recall[:-1])
    best_threshold_recall = thresholds[best_idx_recall]

    print(f"Best threshold for {model} f_recall: {best_threshold_recall}")

    # Optimizing f_2
    b = 2
    f_2 = (1 + b**2) * precision[:-1] * recall[:-1] / (b**2 * precision[:-1] + recall[:-1] + 1e-8) # 1e-8 to avoid zero division

    best_idx_2 = np.argmax(f_2)
    best_threshold_2 = thresholds[best_idx_2]

    print(f"Best threshold for {model} f_2: {best_threshold_2}")

    # Optimizing f1-score
    b = 1
    f_1 = (1 + b**2) * precision[:-1] * recall[:-1] / (b**2 * precision[:-1] + recall[:-1] + 1e-8) # 1e-8 to avoid zero division

    best_idx_1 = np.argmax(f_1)
    best_threshold_1 = thresholds[best_idx_1]

    print(f"Best threshold for {model} f_1: {best_threshold_1}")