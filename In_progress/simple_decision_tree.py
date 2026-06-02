#!/usr/local/anaconda3-2024.10-1/bin/python3
# Edited: 30/4/2026

# Import libraries
import pandas as pd
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

path = ""
df = pd.read_csv(path + 'feature_table_attempt_60D.csv').drop(columns=["patient_id"])

# ---- Split into train/test ---- #
# Random shuffling the rows
df_shuffle = df.sample(frac=1).reset_index(drop=True)
# Find index position for 80 percent of entries
cut_80 = int(len(df_shuffle)*0.8)
df_train = df_shuffle[:cut_80]
df_test = df_shuffle[cut_80:]

# One hot encodings, Diagnosis is categorical
one_hot_train = pd.get_dummies(df_train, dtype = int)
one_hot_test = pd.get_dummies(df_train, dtype = int)

# ---- Train simple decision tree ---- #
X_train = one_hot_train.drop(columns="mtx_fail")
y_train = one_hot_train["mtx_fail"]

# Create Decision Tree classifer object
criterion = "entropy"  # entropy, gini, or log_loss
max_depth = 6
clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)


#Predict the response for test dataset
X_test = one_hot_test.drop(columns="mtx_fail")
y_test = one_hot_test["mtx_fail"]
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot it
plt.figure(figsize=(22,8))
plot_tree(clf, label="root", filled=True, feature_names=list(X_train.columns), fontsize=6)
plt.text(x=0.8, y=0.9, s=f"Accuracy: {accuracy}")
plt.title(f"Simple decision tree 60D ({criterion}, depth: {max_depth})")
plt.tight_layout()
plt.savefig(f"simple_tree_60D_{criterion}_{max_depth}.png")