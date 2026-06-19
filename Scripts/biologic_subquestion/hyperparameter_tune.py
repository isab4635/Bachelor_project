#!/usr/local/anaconda3-2024.10-1/bin/python3
# Hyperparameter tunning for biologic_added model

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

# Settings
path = "../../data/"
outdir = "../../results/bio_models/"
label = 'biologic_added'

# Load train and test data scaled
df_train_sc = pd.read_csv(path + 'scaled_train_feature_table.csv')
df_test_sc = pd.read_csv(path + 'scaled_test_feature_table.csv')

if label == "mtx_stopped":
    df_train_sc.drop(columns=["biologic_added"], inplace=True)
    df_test_sc.drop(columns=["biologic_added"], inplace=True)
elif label == "biologic_added":
    df_train_sc.drop(columns=["mtx_stopped"], inplace=True)
    df_test_sc.drop(columns=["mtx_stopped"], inplace=True)
else:
    raise ValueError("Label must be either 'mtx_stopped' or 'biologic_added'")

X_train_sc = df_train_sc.drop(columns=[label])
y_train_sc = df_train_sc[label]

X_test_sc = df_test_sc.drop(columns=[label])
y_test_sc = df_test_sc[label]

# Load train and test data unscaled
df_train = pd.read_csv(path + 'train_filled_feature_table.csv')
df_test = pd.read_csv(path + 'test_filled_feature_table.csv')

df_train.drop(columns=["mtx_stopped"], inplace=True)
df_test.drop(columns=["mtx_stopped"], inplace=True)
X_train = df_train.drop(columns=["biologic_added"])
y_train = df_train["biologic_added"]

X_test = df_test.drop(columns=["biologic_added"])
y_test = df_test["biologic_added"]

#Checking if y scaled and unscaled the same
if y_train_sc.equals(y_train):
      print("labels are in order")
else:
      raise ValueError("scaled and unscaled label training values unmatched")

if y_train_sc.equals(y_train):
      print("labels are in order")
else:
      raise ValueError("scaled and unscaled label testing values unmatched")

#Improvements to the baseline model

#redundant features
X_train.drop(columns = ["SDAI", "CDAI"], inplace=True)
X_test.drop(columns = ["SDAI", "CDAI"], inplace=True)
X_train_sc.drop(columns = ["SDAI", "CDAI"], inplace=True)
X_test_sc.drop(columns = ["SDAI", "CDAI"], inplace=True)

#Improvements to the baseline model

#handling class imbalance
#https://www.geeksforgeeks.org/machine-learning/how-does-the-classweight-parameter-in-scikit-learn-work/

random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

decision_tree_model = DecisionTreeClassifier(random_state=42, class_weight = "balanced")
decision_tree_model.fit(X_train, y_train)

logistic_model = LogisticRegression(random_state=42, class_weight = "balanced")
logistic_model.fit(X_train_sc, y_train)

gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train_sc, y_train)

# Define parameters for random search
param_dist_logistic = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100]
}

param_dist_rf = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"]
}

param_dist_dt = {
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"]
}

param_dist_gbc = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 10, None],
    "n_estimators": [100, 200, 300, 500],
    "min_samples_leaf": [1, 5, 10],
    "subsample": [0.8, 1.0]
}


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

models = [logistic_model, random_forest_model, decision_tree_model, gbc]
params = [param_dist_logistic, param_dist_rf, param_dist_dt, param_dist_gbc]

# Get predictions for each model
for i, model in enumerate(['Logistic', 'RandomForest', 'DecisionTree', 'GradientBoosting']):
    print(f"Performing random search for {model}...")
    search = RandomizedSearchCV(
    estimator=models[i],
    param_distributions=params[i],
    scoring="average_precision",
    random_state=42
    )

    if model in ['Logistic', 'GradientBoosting']:
             search.fit(X_train_sc, y_train)

             print("Best parameters:")
             print(search.best_params_)

             print("\nBest CV recall:")
             print(search.best_score_)

             best_model = search.best_estimator_

             y_pred = best_model.predict(X_test_sc)

             print("\nTest set performance:")
             print(classification_report(y_test, y_pred))

             y_pred_prob = best_model.predict_proba(X_test_sc)[:, 1]

    else:
             search.fit(X_train, y_train)

             print("Best parameters:")
             print(search.best_params_)

             print("\nBest CV recall:")
             print(search.best_score_)

             best_model = search.best_estimator_

             y_pred = best_model.predict(X_test)

             print("\nTest set performance:")
             print(classification_report(y_test, y_pred))

             y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    # Plot ROC curve and compute AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axs[0].plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')
    # Plot PR curve and compute AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    ap = average_precision_score(y_test, y_pred_prob)
    axs[1].plot(recall, precision, label=f'{model} (AP = {ap:.2f})')


axs[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
axs[1].hlines(y_test.mean(), 0, 1, colors='gray', linestyles='--', label=f"Baseline (AP = {y_test.mean():.2f})")

axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC Curves for Multiple Models')
axs[0].legend()
axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')
axs[1].set_title('PR Curves for Multiple Models')
axs[1].legend()
plt.savefig(outdir + "best_tune_roc_pr.png")
plt.close()
