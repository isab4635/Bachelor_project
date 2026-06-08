#!/usr/local/anaconda3-2024.10-1/bin/python3
# From https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/
# Script creating the baseline models for the biologic subquestion, and plotting ROC and precision-recall curves. Also prints confusion matrix and other evaluation metrics for each model.

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
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
plt.savefig(outdir + "baseline_roc.png")
plt.close()

plt.figure(figsize=(7, 5))

# Plot precision-recall curves and calculate average precision scores
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
plt.savefig(outdir + "baseline_pr.png")
plt.close()

# Print evaluation metrics
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

print(f"Baseline model evaluation metrics: ")
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