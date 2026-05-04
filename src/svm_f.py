#!/usr/bin/env python3
"""
svm_f.py — fixed-parameter LOSO check on the full 65-subject set.

Purpose:
- Use one fixed (C, gamma) pair across all folds
- Evaluate on AD 1–36 and control 37–65 (FTD excluded)
- Compare fixed-parameter performance against tuning outputs

To run: cd src && python3 svm_f.py
"""
from os import getcwd, execv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Working directories
cwd = Path(getcwd())
data_root = cwd.parent / "processed"
alz_root = data_root / "alz"
con_root = data_root / "con"

# Check if processed dataset exists, if not run dsprocess.py
if not (alz_root.exists() and con_root.exists()):
    print("Processed dataset does not exist.\nNow running dsprocess.py...\n")
    execv("./dsprocess.py", [" "])

# Full-dataset ranges (FTD 66–88 excluded)
alz_index = (1, 36)
con_index = (37, 65)

ad_ids = list(range(alz_index[0], alz_index[1] + 1))
con_ids = list(range(con_index[0], con_index[1] + 1))
eval_ids = ad_ids + con_ids

# Toggle preprocessing for quick A/B checks against svm.py
USE_SCALER = False

acc = 0
total_rec = 0
y_true_all: list[int] = []
y_pred_all: list[int] = []

# LOSO over all 65 subjects: hold out 1, train on the remaining 64.
for skip in eval_ids:
    skip_file = "sub_" + str(skip).zfill(3) + "_rbp.csv"
    skip_root = alz_root if skip in ad_ids else con_root

    # Training AD rows (excluding held-out subject)
    alz_list = []
    for idx in ad_ids:
        if idx == skip:
            continue
        sub_file = "sub_" + str(idx).zfill(3) + "_rbp.csv"
        alz_list.append(pd.read_csv(alz_root / sub_file, sep=","))

    alz_data = pd.concat(alz_list).to_numpy(dtype=np.float64)
    del alz_list

    # Training control rows (excluding held-out subject)
    con_list = []
    for idx in con_ids:
        if idx == skip:
            continue
        sub_file = "sub_" + str(idx).zfill(3) + "_rbp.csv"
        con_list.append(pd.read_csv(con_root / sub_file, sep=","))

    # Held-out subject: all feature rows from their file become the test set.
    test = pd.read_csv(skip_root / skip_file, sep=",").to_numpy(dtype=np.float64)
    con_data = pd.concat(con_list).to_numpy(dtype=np.float64)
    del con_list

    count_a_recs = alz_data.shape[0]
    count_c_recs = con_data.shape[0]
    total_data = np.concatenate([con_data, alz_data], axis=0)
    del alz_data, con_data

    # Labels match total_data row order: controls (0) then Alzheimer's (1).
    bin_labels = [0] * count_c_recs + [1] * count_a_recs

    # True class for held-out subject rows
    expected = 1 if skip in ad_ids else 0

    if USE_SCALER:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(total_data)
        X_test = scaler.transform(test)
    else:
        X_train = total_data
        X_test = test

    #SVM Implementation, tune hyperameters here
    clf = SVC(kernel="poly", degree=3, coef0=2.0, C=1, gamma="scale", class_weight="balanced")
    clf.fit(X_train, bin_labels)

    print(f"Testing subject {skip}'s data...")
    lab_arr = clf.predict(X_test)

    # Row-level accuracy: count test predictions matching the subject's true label.
    total_rec += len(lab_arr)
    acc += lab_arr.tolist().count(expected)
    y_true_all.extend([expected] * len(lab_arr))
    y_pred_all.extend(lab_arr.tolist())

# Accuracy: overall fraction of correct predictions across all rows.
accuracy = acc / total_rec
print(f"Accuracy Rate: {accuracy * 100}%")

# Balanced Accuracy: average of recall for each class (helps with imbalance).
bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
print(f"Balanced Accuracy: {bal_acc * 100}%")

# Precision (AD=1): of predicted AD rows, how many are truly AD.
precision_ad = precision_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
print(f"Precision (AD=1): {precision_ad * 100}%")

# Recall (AD=1): of all true AD rows, how many were detected.
recall_ad = recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
print(f"Recall (AD=1): {recall_ad * 100}%")

# F1 (AD=1): harmonic mean of AD precision and AD recall.
f1_ad = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
print(f"F1 (AD=1): {f1_ad * 100}%")

# Classification report: per-class precision/recall/F1 and support.
print("\nClassification Report (Row-Level):")
print(classification_report(y_true_all, y_pred_all, target_names=["control", "alzheimers"]))

# Run history (full 65-subject LOSO):
# - C=0.1, gamma=0.001, class_weight="balanced" -> Accuracy Rate: 64.20116112493403%
# - C=0.11, gamma=0.0012, class_weight="balanced" -> Accuracy Rate:  64.81942245344192%
# - C=1.0, gamma='scale', class_weight="balanced" -> Accuracy Rate: 68.12184272034985%
# - C=0.1, gamma='scale', class_weight="balanced" -> Accuracy Rate: 68.31033702782176%
# - C=1, gamma=0.2, class_weight="balanced" -> Accuracy Rate: 68.12184272034985%
#All the prior runs were with standard scaler, now we are using without it.
# Kernel: rbf, C=1, gamma='scale', class_weight="balanced" -> Accuracy Rate: 68.30656714167233%
# Kernel: poly, degree: 3, C: 1.0, gamma: scale, class_weight: balanced -> Accuracy Rate: 68.69486541506447%
#Kernel: poly, degree: 3, C: 0.1, gamma: scale, class_weight: balanced -> Accuracy Rate: 68.2839478247757%
#Kernel: poly, degree: 3, C: 0.1, gamma: scale, class_weight: balanced -> Accuracy Rate: 59.71122672095303% with scaler
#Kernel: poly, degree: 3, C: 0.05, gamma: 0.02, coef0: 2.0, class_weight: balanced -> Accuracy Rate: 56.303249641860816%
#Kernel: poly, degree: 2, C: 0.1, gamma: scale, coef0: 2.0, class_weight: balanced -> Accuracy Rate: 67.62044786247455%
#Kernel: rbf, C: 0.1, gamma: scale, class_weight: balanced -> Accuracy Rate: 66.22182010103295%
#Kernel: poly, degree: 3, C: 1, gamma: scale, coef0: 2.0, class_weight: balanced, Accuracy Rate: 68.7174847319611%
#Kernel: rbf, C: 50, gamma: scale, class_weight: balanced -> Accuracy Rate: 62.968408354067705%
#Kernel: rbf, C: 50, gamma: 0.001, class_weight: balanced -> Accuracy Rate: 68.67601598431727%,
#Kernel: poly, degree: 4, C: 1, gamma: scale, coef0: 2.0, class_weight: balanced -> Accuracy Rate: 68.69109552891503%
#Kernel: poly, degree: 2, C: 1, gamma: scale, coef0: 2.0, class_weight: balanced -> Accuracy Rate: 68.46490235994874%