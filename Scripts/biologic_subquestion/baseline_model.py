#!/usr/local/anaconda3-2024.10-1/bin/python3
# From https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/
# Baseline models for biologic_added and their evaluation using ROC and PR curves, as well as confusion matrix and other metrics. This is a simple baseline to compare against more complex models later on.

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

# Settings
label = "biologic_added"
path = "../../data/"
outdir = "../../results/bio_models/"

# Load train and test data
df_train = pd.read_csv(path + 'train_filled_feature_table.csv')
df_test = pd.read_csv(path + 'test_filled_feature_table.csv')

if label == "mtx_stopped":
    df_train.drop(columns=["biologic_added"], inplace=True)
    df_test.drop(columns=["biologic_added"], inplace=True)
elif label == "biologic_added":
    df_train.drop(columns=["mtx_stopped"], inplace=True)
    df_test.drop(columns=["mtx_stopped"], inplace=True)
else:
    raise ValueError("Label must be either 'mtx_stopped' or 'biologic_added'")

X_train = df_train.drop(columns=[label])
y_train = df_train[label]

X_test = df_test.drop(columns=[label])
y_test = df_test[label]

# Training models (logistic and RF and simple decision tree)
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Get predictions
y_pred_logistic = logistic_model.predict_proba(X_test)[:, 1]
y_pred_rf = random_forest_model.predict_proba(X_test)[:, 1]
y_pred_dt = decision_tree_model.predict_proba(X_test)[:, 1]

# Create dataframe and plot ROCs
test_df = pd.DataFrame(
    {'True': y_test, 'Logistic': y_pred_logistic, 'RandomForest': y_pred_rf, 'DecisionTree': y_pred_dt})

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, precision_recall_curve, average_precision_score

# Print additional evalution metrics
for model in ['Logistic', 'RandomForest', 'DecisionTree']:
    # Binary predictions for confusion matrix and precision/recall
    preds = (test_df[model] >= 0.5).astype(int)

    print(f"Model: {model}")
    print(confusion_matrix(y_test, preds))
    print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1score = f1_score(y_test, preds)
    b = 2
    f_2 = (1 + b**2) * precision * recall / (b**2 * precision + recall + 1e-8) # 1e-8 to avoid zero division
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1score:.2f}, F2-score: {f_2:.2f}")

# Print PR-curve and ROC-curve
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for model in ['Logistic', 'RandomForest', 'DecisionTree']:
    # Plot ROC curve and compute AUC
    fpr, tpr, _ = roc_curve(test_df['True'], test_df[model])
    roc_auc = auc(fpr, tpr)
    axs[0].plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')
    # Plot PR curve and compute AUC
    precision, recall, _ = precision_recall_curve(test_df['True'], test_df[model])
    pr_auc = auc(recall, precision)
    print(f"PR-AUC ({model}): {pr_auc:.2f}")
    ap = average_precision_score(test_df['True'], test_df[model])
    axs[1].plot(recall, precision, label=f'{model} (AP = {ap:.2f})')


axs[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
axs[1].hlines(test_df['True'].mean(), 0, 1, colors='gray', linestyles='--', label=f"Baseline (AP = {test_df['True'].mean():.2f})")

axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC Curves for Multiple Models')
axs[0].legend()
axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')
axs[1].set_title('PR Curves for Multiple Models')
axs[1].legend()
plt.savefig(outdir + "baseline_roc_pr.png")
plt.close()