#!/usr/local/anaconda3-2024.10-1/bin/python3
# Script creating the improved models for the biologic subquestion, and plotting ROC and precision-recall curves. Also prints confusion matrix and other evaluation metrics for each model.
# Optimisation: removing redundant features, log-scaling CRP, standardizing data, and handling class imbalance with class weights.

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from math import log, log1p

# Settings
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
print(f"columns: {X_train.columns}")

#redundant features
#if SDAI AND CDAI still in the futures
if "SDAI" or "CDAI" in X_train.columns:
    print("Removing redundant features: ")
    X_train.drop(columns = ["SDAI", "CDAI"], inplace=True)
    X_test.drop(columns = ["SDAI", "CDAI"], inplace=True)
else:
    print("No redundant features")

#handling class imbalance
#https://www.geeksforgeeks.org/machine-learning/how-does-the-classweight-parameter-in-scikit-learn-work/

#log-csaling CRP
X_train["CRP"] = X_train["CRP"].apply(lambda x: log1p(x))
X_test["CRP"] = X_test["CRP"].apply(lambda x: log1p(x))

#normalizing and scaling data (standardizing)
col_to_change = [col for col in X_train.columns if col not in ["IgM_RF", "Anti_CCP"]]

X_mean = X_train[col_to_change].mean()
X_std = X_train[col_to_change].std()
X_train[col_to_change] = (X_train[col_to_change] - X_mean) / X_std
X_test[col_to_change] = (X_test[col_to_change] - X_mean) / X_std

# Training models (logistic and RF and simple decision tree)
logistic_model = LogisticRegression(random_state=42, class_weight = "balanced")
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(random_state=42, class_weight = "balanced")
random_forest_model.fit(X_train, y_train)

decision_tree_model = DecisionTreeClassifier(random_state=42, class_weight = "balanced")
decision_tree_model.fit(X_train, y_train)


# Get predictions
y_pred_logistic = logistic_model.predict_proba(X_test)[:, 1]
y_pred_rf = random_forest_model.predict_proba(X_test)[:, 1]
y_pred_dt = decision_tree_model.predict_proba(X_test)[:, 1]


# Create dataframe and plot ROCs
test_df = pd.DataFrame(
    {'True': y_test, 'Logistic': y_pred_logistic, 'RandomForest': y_pred_rf, 'DecisionTree': y_pred_dt})

plt.figure(figsize=(7, 5))

for model in ['Logistic', 'RandomForest', 'DecisionTree']:
    fpr, tpr, _ = roc_curve(test_df['True'], test_df[model])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiple Models')
plt.legend()
plt.savefig(outdir + "model_roc.png")
plt.close()

plt.figure(figsize=(7, 5))

from sklearn.metrics import precision_recall_curve, average_precision_score

prevalence = test_df['True'].mean()

for model in ['Logistic', 'RandomForest', 'DecisionTree']:
     precision, recall, thresholds = precision_recall_curve(test_df['True'], test_df[model])
     ap = average_precision_score(test_df['True'], test_df[model])
     plt.plot(recall, precision, label=f"{model} (AP = {ap:.3f})")
plt.hlines(prevalence, 0, 1, colors='gray', linestyles='--',
           label=f"Baseline = {prevalence:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(outdir + "model_pr.png")
plt.close()

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

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
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1score:.2f}")