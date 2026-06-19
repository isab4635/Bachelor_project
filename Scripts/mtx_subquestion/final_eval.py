#!/usr/local/anaconda3-2024.10-1/bin/python3
# Final evaluation on logistic regression and random forest models

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


# Settings
label = "mtx_stopped"
path = "../../data/"
outdir = "../../results/final_mtx_model/"

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

# Drop features strongly correlated with DAS
df_train.drop(columns=["CDAI", "SDAI"], inplace=True)
df_test.drop(columns=["CDAI", "SDAI"], inplace=True)

# Split into X and y
X_train = df_train.drop(columns=[label])
y_train = df_train[label]

X_test = df_test.drop(columns=[label])
y_test = df_test[label]

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
    'mtx_start_year': 'MTX start year',
    'IgM_RF': 'RF',
    'Anti_CCP': 'Anti-CCP',
    'Koen': 'Sex'
}


# ---- Training models ---- #
logistic_model = LogisticRegression(random_state=42, C=0.1, class_weight="balanced")
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_leaf=10, max_depth=5, criterion="entropy")
random_forest_model.fit(X_train, y_train)

# ---- Feature importance plot for logistic regression ---- #
importances = logistic_model.coef_[0]
# Sort feature importances in descending order
indices = np.argsort(np.abs(importances))[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]
pretty_names = [pretty_labels[name] for name in names]

# Create plot
plt.figure(figsize=(7, 5))
plt.title("Feature Coefficients for Logistic Regression")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), pretty_names, rotation=450)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.savefig(outdir + "feature_importances_logistic.png")
plt.close()

# ---- Feature importance plot for random forest ---- #
importances = random_forest_model.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]
pretty_names = [pretty_labels[name] for name in names]

# Create plot
plt.figure(figsize=(7, 5))
plt.title("Feature Importances for Random Forest")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), pretty_names, rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(outdir + "feature_importances_random_forest.png")
plt.close()


# ---- Get probabilities ---- #
# Get probabilities
y_pred_logistic = logistic_model.predict_proba(X_test)[:, 1]
y_pred_rf = random_forest_model.predict_proba(X_test)[:, 1]

# Map true labels to descriptive strings for better visualization
y_test_label = y_test.map({0: "MTX continued", 1: "MTX stopped"})

# Create a DataFrame to store predictions and true labels
test_df = pd.DataFrame(
    {'True': y_test, 'True_label': y_test_label,'Logistic': y_pred_logistic, 'RandomForest': y_pred_rf})


# ---- Swarm plot of predicted probabilities ---- #
fig, axs = plt.subplots(1, 2, figsize=(4, 6))

sb.stripplot(
    y='Logistic',
    hue='True_label',
    data=test_df,
    jitter=0.25,
    alpha=0.7,
    ax=axs[0]
)
axs[0].set_title("Logistic Regression")
axs[0].set_ylabel("Predicted Probability")

sb.stripplot(
    y='RandomForest',
    hue='True_label',
    data=test_df,
    jitter=0.25,
    alpha=0.7,
    ax=axs[1]
)
axs[1].set_title("Random Forest")
axs[1].set_ylabel("Predicted Probability")

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend().remove()
axs[1].legend().remove()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.2, 0.5), title='MTX Status')
fig.suptitle("Swarm plots of predicted probabilities")
plt.tight_layout()
plt.savefig(outdir + "swarm_plots.png", dpi=300, bbox_inches='tight')
plt.close()


# ---- Confusion matrix for logistic regression at 0.5 threshold ---- #
thresholds = {'Logistic': [0.67, 0.52, 0.38], 'RandomForest': [0.53, 0.33, 0.22]}

for model in ['Logistic', 'RandomForest']:
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    # Binary predictions for confusion matrix and precision/recall
    for i, threshold in enumerate(thresholds[model]):
        y_pred = (test_df[model] >= threshold).astype(int)

        # Precision, recall, F1-score
        print(f"Model: {model} (threshold={threshold})")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        b = 0.5
        f_05 = (1 + b**2) * precision * recall / (b**2 * precision + recall + 1e-8) # 1e-8 to avoid zero division

        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1score:.2f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        axs[i].set_title(f"{model} (threshold={threshold})")
        sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[i])
        axs[i].set_xlabel("Predicted Label")
        axs[i].set_ylabel("True Label")

    plt.suptitle("Confusion matrices")
    plt.tight_layout()
    plt.savefig(outdir + f"confusion_matrices_{model}.png", dpi=300, bbox_inches='tight')
    plt.close()