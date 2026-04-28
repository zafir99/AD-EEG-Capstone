#!/usr/bin/env python3
"""
finetune.py

Compact SVM fine-tuning script:
- Uses a fixed subset of 10 AD subjects and 10 control subjects
- Outer evaluation: leave-one-subject-out (LOSO) over those 20 subjects
- Inner tuning: GridSearchCV with Pipeline(StandardScaler + SVC)
- Current mode runs a wide 5x5 grid over C and gamma
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    # Preferred: keeps subject-group isolation while preserving class balance.
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

# =========================
# Fine-tuning search space
# =========================
# Hyperparameter candidates used by GridSearchCV.
# Keep these as plain Python lists so they are easy to edit between runs.
# NOTE: gamma may be either numeric (float/int) OR the strings "scale"/"auto".
#       If you include string gamma options, keep the tracking list types below as str.
# C values to test
GRID_C_VALUES = [20, 25, 30, 35, 40, 45, 50]
# Gamma values to test
GRID_GAMMA_VALUES = [0.001]
# Degree values for polynomial kernel
# GRID_DEGREES = [2, 3]
# FIXED_DEGREE = 2
# Bias term values used by polynomial kernel
# GRID_COEF0_VALUES = [0, 0.5, 1, 2]
# FIXED_COEF0 = 2.0
FIXED_CLASS_WEIGHT = "balanced"
# Fixed kernel config for this experiment
FIXED_KERNEL = "rbf"
# If you want to grid-search class weights later, define candidates like:
# GRID_CLASS_WEIGHTS = ["balanced", None, {0: 1.0, 1: 1.0}]

def load_subject_csv(folder: Path, subject_id: int) -> np.ndarray:
    """Load one subject's processed RBP feature matrix."""
    file_name = f"sub_{subject_id:03d}_rbp.csv"
    matrix = pd.read_csv(folder / file_name, sep=",").to_numpy(dtype=np.float64)
    return matrix


def main() -> None:
    # -----------------------------
    # 1) CLI arguments / run config
    # -----------------------------
    parser = argparse.ArgumentParser(description="LOSO + GridSearch fine-tuning on a 20-subject subset.")
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=3,
        help="Inner CV splits for GridSearchCV (lower values reduce single-class fold warnings).",
    )
    parser.add_argument(
        "--show-cv-warnings",
        action="store_true",
        help="Show sklearn fold warnings (hidden by default).",
    )
    args = parser.parse_args()

    # ---------------------------------------
    # 2) Locate dataset folders (repo-relative)
    # ---------------------------------------
    repo_root = Path(__file__).resolve().parent.parent
    processed_root = repo_root / "processed"
    alz_root = processed_root / "alz"
    con_root = processed_root / "con"

    if not (alz_root.exists() and con_root.exists()):
        raise FileNotFoundError("Missing processed/alz or processed/con. Run dsprocess.py first.")

    # ------------------------------------------
    # 3) Fixed subset for fast/reproducible runs
    # ------------------------------------------
    ad_ids = list(range(1, 11))      # 10 AD
    con_ids = list(range(37, 47))    # 10 controls

    # (subject_id, label, folder): label 1 = AD, 0 = control
    subjects: list[tuple[int, int, Path]] = []
    subjects.extend((sid, 1, alz_root) for sid in ad_ids)
    subjects.extend((sid, 0, con_root) for sid in con_ids)

    # Preload each subject once to avoid repeated disk I/O inside LOSO folds.
    subject_data: dict[int, np.ndarray] = {}
    subject_label: dict[int, int] = {}
    for sid, label, folder in subjects:
        subject_data[sid] = load_subject_csv(folder, sid)
        subject_label[sid] = label

    # Collect final predictions across all outer LOSO folds.
    all_true_rows: list[int] = []
    all_pred_rows: list[int] = []
    # Best-hyperparameter tracking across outer folds.
    # C is always numeric, gamma is stored as string so mixed values
    # (e.g., 0.01 and "scale") can be counted in one structure.
    best_c_values: list[float] = []
    best_gamma_values: list[str] = []
    best_param_pairs: list[tuple[float, str]] = []

    print("=== Fine-tune Setup ===")
    print(f"Subjects used (20 total): AD={ad_ids}, Control={con_ids}")
    print(f"Grid C values: {GRID_C_VALUES}")
    print(f"Grid gamma values: {GRID_GAMMA_VALUES}")
    print(f"Fixed class_weight: {FIXED_CLASS_WEIGHT}")
    print("")

    # -------------------------------------------------------------
    # 4) Outer CV: LOSO over 20 subjects (hold out one subject each)
    # -------------------------------------------------------------
    for holdout_sid, holdout_label, _ in subjects:
        # Test set = every row from the held-out subject.
        X_test = subject_data[holdout_sid]
        y_test = np.full(X_test.shape[0], holdout_label, dtype=int)

        # Build train set from the remaining 19 subjects.
        X_train_chunks: list[np.ndarray] = []
        y_train_chunks: list[np.ndarray] = []
        # groups_chunks tracks subject IDs for GroupKFold inside GridSearchCV.
        groups_chunks: list[np.ndarray] = []

        for train_sid, _, _ in subjects:
            if train_sid == holdout_sid:
                continue
            data = subject_data[train_sid]
            lbl = subject_label[train_sid]
            X_train_chunks.append(data)
            y_train_chunks.append(np.full(data.shape[0], lbl, dtype=int))
            groups_chunks.append(np.full(data.shape[0], train_sid, dtype=int))

        X_train = np.concatenate(X_train_chunks, axis=0)
        y_train = np.concatenate(y_train_chunks, axis=0)
        groups_train = np.concatenate(groups_chunks, axis=0)

        # Inner CV cannot exceed available training groups.
        n_groups = len(np.unique(groups_train))
        n_splits = min(args.inner_splits, n_groups)
        if n_splits < 2:
            raise ValueError("Need at least 2 groups for inner GroupKFold.")

        # Pipeline keeps preprocessing tied to model fitting per split.
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel=FIXED_KERNEL,
                        class_weight=FIXED_CLASS_WEIGHT,
                    ),
                ),
            ]
        )

        # The parameter grid tells GridSearchCV what to tune 
        # Add/remove keys here when changing the tuning scenario.
        param_grid = {
            "svc__C": GRID_C_VALUES,
            "svc__gamma": GRID_GAMMA_VALUES,
            # "svc__degree": GRID_DEGREES,
            # "svc__coef0": GRID_COEF0_VALUES,
            # "svc__class_weight": GRID_CLASS_WEIGHTS,
        }

        # -------------------------------------------------------------------------
        # 5) Inner CV: prefer StratifiedGroupKFold to avoid single-class val folds
        # -------------------------------------------------------------------------
        if StratifiedGroupKFold is not None:
            inner_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            # Fallback for older scikit-learn versions.
            inner_cv = GroupKFold(n_splits=n_splits)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )
        if args.show_cv_warnings:
            search.fit(X_train, y_train, groups=groups_train)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="y_pred contains classes not in y_true",
                    category=UserWarning,
                )
                search.fit(X_train, y_train, groups=groups_train)
        # Store best hyperparameters selected for this outer fold.
        # Keep gamma as str to safely support numeric + keyword gamma values.
        best_c = float(search.best_params_["svc__C"])
        best_gamma = str(search.best_params_["svc__gamma"])
        best_c_values.append(best_c)
        best_gamma_values.append(best_gamma)
        best_param_pairs.append((best_c, best_gamma))

        row_preds = search.predict(X_test)
        all_true_rows.extend(y_test.tolist())
        all_pred_rows.extend(row_preds.tolist())

        fold_row_acc = accuracy_score(y_test, row_preds)
        print(
            f"Holdout subject {holdout_sid:03d} | best={search.best_params_} "
            f"| row_acc={fold_row_acc:.4f}"
        )

    #Count the frequency of the best parameters across all folds
    c_counter = Counter(best_c_values)
    gamma_counter = Counter(best_gamma_values)
    pair_counter = Counter(best_param_pairs)
    top_c, top_c_count = c_counter.most_common(1)[0]
    top_gamma, top_gamma_count = gamma_counter.most_common(1)[0]
    top_pair, top_pair_count = pair_counter.most_common(1)[0]
    # degree_counter = Counter(best_degree_values)
    # coef0_counter = Counter(best_coef0_values)
    # class_weight_counter = Counter(best_class_weight_values)
    # top_degree, top_degree_count = degree_counter.most_common(1)[0]
    # top_coef0, top_coef0_count = coef0_counter.most_common(1)[0]
    # top_class_weight, top_class_weight_count = class_weight_counter.most_common(1)[0]

    print("\n=== Best Parameter Frequency (Across 20 LOSO Folds) ===")
    print(f"C counts: {dict(sorted(c_counter.items()))}")
    print(f"Gamma counts: {dict(gamma_counter)}")
    print(f"Top C: {top_c} ({top_c_count}/20 folds)")
    print(f"Top gamma: {top_gamma} ({top_gamma_count}/20 folds)")
    # print(f"Degree counts: {dict(sorted(degree_counter.items()))}")
    # print(f"coef0 counts: {dict(sorted(coef0_counter.items()))}")
    # print(f"Class-weight counts: {dict(class_weight_counter)}")
    # print(f"Top degree: {top_degree} ({top_degree_count}/20 folds)")
    # print(f"Top coef0: {top_coef0} ({top_coef0_count}/20 folds)")
    # print(f"Top class_weight: {top_class_weight} ({top_class_weight_count}/20 folds)")
    print(
        f"Top (C, gamma): ({top_pair[0]}, {top_pair[1]}) "
        f"({top_pair_count}/20 folds)"
    )


    # -----------------------------------
    # 6) Aggregate and print final metrics
    # -----------------------------------
    print("\n=== Final Metrics (Row-Level) ===")
    print(f"Accuracy:          {accuracy_score(all_true_rows, all_pred_rows):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(all_true_rows, all_pred_rows):.4f}")
    print(f"Precision (AD=1):  {precision_score(all_true_rows, all_pred_rows, pos_label=1):.4f}")
    print(f"Recall (AD=1):     {recall_score(all_true_rows, all_pred_rows, pos_label=1):.4f}")
    print(f"F1 (AD=1):         {f1_score(all_true_rows, all_pred_rows, pos_label=1):.4f}")
    print("\nClassification Report (Row-Level):")
    print(classification_report(all_true_rows, all_pred_rows, target_names=["control", "alzheimers"]))


if __name__ == "__main__":
    main()

#First run of finetuning 
#GRID_C_VALUES = [0.1, 1, 10]
#GRID_GAMMA_VALUES = [0.001, 0.01, 0.1]
#Best parameters: C = 0.1, gamma = 0.001
#Accuracy: 0.4852
#Balanced Accuracy: 0.5165

#Second run of finetuning 
#GRID_C_VALUES = [0.03, 0.1, 0.3]
#GRID_GAMMA_VALUES = [0.0003, 0.001, 0.003, 'scale']
#Best parameters: 0.3, 0.0003
#Accuracy: 0.4804
#Balanced Accuracy: 0.5126

#Third run of finetuning 
#GRID_C_VALUES = [0.05, 0.1, 0.2]
#GRID_GAMMA_VALUES = [0.0005, 0.001, 0.002]
#Best parameters: 0.1, 0.0005
#Accuracy: 0.4540
#Balanced Accuracy: (add from run output)

#Fourth run of finetuning (I'll be testing class weights)
#C: 0.1
#gamma: 0.001
#GRID_CLASS_WEIGHTS = [None, "balanced", {0: 1.2, 1: 1.0}, {0: 1.5, 1: 1.0}, {0: 2.0, 1: 1.0}]
#Most common class_weight: "balanced" (9/20 folds)
#Accuracy: 0.1046
#Balanced Accuracy: 0.0964

#Fifth run of finetuning (class weights, updated set, used balanced accuracy as the main metric of GRIDSV)
#C: 0.1
#gamma: 0.001
#GRID_CLASS_WEIGHTS = [{0: 1.0, 1: 1.0}, "balanced", None, {0: 1.0, 1: 1.5}, {0: 1.5, 1: 1.0}]
#Most common class_weight: "balanced" (15/20 folds)
#Accuracy: 0.5950
#Balanced Accuracy: 0.6220

#Sixth run of finetuning (3-way grid: C, gamma, class_weight)
#GRID_C_VALUES = [0.09, 0.1, 0.11]
#GRID_GAMMA_VALUES = [0.0008, 0.001, 0.0012]
#GRID_CLASS_WEIGHTS = ["balanced", {0: 1.0, 1: 1.0}, {0: 1.2, 1: 1.0}]
#Top C: 0.11 (14/20 folds)
#Top gamma: 0.0012 (12/20 folds)
#Most common class_weight: "balanced" (17/20 folds)
#Top (C, gamma, class_weight): (0.11, 0.0012, "balanced") (11/20 folds)
#Accuracy: 0.5420
#Balanced Accuracy: 0.5650

#Seventh run setup (wide C/gamma sweep, class_weight fixed)
#GRID_C_VALUES = [0.01, 0.1, 1, 10, 100]
#GRID_GAMMA_VALUES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#FIXED_CLASS_WEIGHT = "balanced"
#Just going to straight up copy the results because there is too much to look into
"""
Holdout subject 001 | best={'svc__C': 0.1, 'svc__gamma': 0.1} | row_acc=0.9966
Holdout subject 002 | best={'svc__C': 10, 'svc__gamma': 0.01} | row_acc=0.3519
Holdout subject 003 | best={'svc__C': 0.1, 'svc__gamma': 0.1} | row_acc=0.1447
Holdout subject 004 | best={'svc__C': 10, 'svc__gamma': 0.001} | row_acc=0.9972
Holdout subject 005 | best={'svc__C': 10, 'svc__gamma': 0.001} | row_acc=0.9352
Holdout subject 006 | best={'svc__C': 100, 'svc__gamma': 0.01} | row_acc=0.1460
Holdout subject 007 | best={'svc__C': 100, 'svc__gamma': 0.01} | row_acc=0.9869
Holdout subject 008 | best={'svc__C': 100, 'svc__gamma': 0.01} | row_acc=0.8405
Holdout subject 009 | best={'svc__C': 10, 'svc__gamma': 0.01} | row_acc=0.2852
Holdout subject 010 | best={'svc__C': 10, 'svc__gamma': 0.0001} | row_acc=0.8732
Holdout subject 037 | best={'svc__C': 0.01, 'svc__gamma': 1e-05} | row_acc=0.0000
Holdout subject 038 | best={'svc__C': 0.01, 'svc__gamma': 0.01} | row_acc=0.0495
Holdout subject 039 | best={'svc__C': 100, 'svc__gamma': 0.01} | row_acc=0.6580
Holdout subject 040 | best={'svc__C': 100, 'svc__gamma': 0.01} | row_acc=0.4948
Holdout subject 041 | best={'svc__C': 0.1, 'svc__gamma': 0.1} | row_acc=0.1244
Holdout subject 042 | best={'svc__C': 0.01, 'svc__gamma': 0.1} | row_acc=0.4625
Holdout subject 043 | best={'svc__C': 0.1, 'svc__gamma': 0.1} | row_acc=0.0993
Holdout subject 044 | best={'svc__C': 0.01, 'svc__gamma': 0.1} | row_acc=0.5604
Holdout subject 045 | best={'svc__C': 0.01, 'svc__gamma': 0.1} | row_acc=0.1085
Holdout subject 046 | best={'svc__C': 100, 'svc__gamma': 0.01} | row_acc=0.8747

=== Best Parameter Frequency (Across 20 LOSO Folds) ===
C counts: {0.01: 5, 0.1: 4, 10.0: 5, 100.0: 6}
Gamma counts: {1e-05: 1, 0.0001: 1, 0.001: 2, 0.01: 9, 0.1: 7}
Top C: 100.0 (6/20 folds)
Top gamma: 0.01 (9/20 folds)
Top (C, gamma): (100.0, 0.01) (6/20 folds)

=== Final Metrics (Row-Level) ===
Accuracy:          0.5113
Balanced Accuracy: 0.5269
Precision (AD=1):  0.4771
Recall (AD=1):     0.7111
F1 (AD=1):         0.5710

Classification Report (Row-Level):
              precision    recall  f1-score   support

     control       0.58      0.34      0.43      4309
  alzheimers       0.48      0.71      0.57      3634

    accuracy                           0.51      7943
   macro avg       0.53      0.53      0.50      7943
weighted avg       0.54      0.51      0.50      7943
"""
#TLDR, 100 is the most common C value but it isn't by a significant margin
#Not every 100, 0.01 pair works
#Might need to test some more with higher C values? As in I have been hovering around 0.1 and as of now obtaining higher accuracy is kinda rough

#Eighth run setup (higher C focus + local gamma)
#GRID_C_VALUES = [10, 50, 100]
#GRID_GAMMA_VALUES = [0.0001, 0.001, 0.01]
#FIXED_CLASS_WEIGHT = "balanced"
#GridSearch scoring = "accuracy"
#Top (C, gamma): (100, 0.01) (7/20 folds)
#Accuracy: 0.5499
#Balanced Accuracy: 0.5636

#Ninth run (mixed C<1 and C>1 test, scoring = accuracy)
#GRID_C_VALUES = [0.1, 0.5, 1, 10, 50]
#GRID_GAMMA_VALUES = [0.0005, 0.001, 0.01, 0.002]
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 50.0 (10/20 folds)
#Top gamma: 0.01 (6/20 folds)
#Top (C, gamma): (50.0, 0.001) (4/20 folds)
#Accuracy: 0.4940
#Balanced Accuracy: 0.5114

#Tenth run??? (Dude I lost track) 
#Based on running svm.py, it seems that the original parameters of C = 1 and
# gamma='scale' seem to get the highest accuracy? I also have heard that automatically
# the gamma would be 0.2 due to there being 5 features. Now going to do a generic test 
# of C values around 1, and using scale, 0.2, and some other gamma values to work with.
#C counts: {0.1: 7, 0.5: 5, 1.0: 3, 5.0: 3, 10.0: 2}
#Gamma counts: {'0.1': 4, '0.01': 15, 'scale': 1}
#Top C: 0.1 (7/20 folds)
#Top gamma: 0.01 (15/20 folds)
#Top (C, gamma): (0.5, 0.01) (5/20 folds)

#"Eleventh" run (poly kernel fixed degree=3, fixed class weight)
#GRID_C_VALUES = [0.01, 0.1, 1, 10]
#GRID_GAMMA_VALUES = ["scale", 0.001, 0.01, 0.05, 0.2]
#FIXED_KERNEL = "poly"
#FIXED_DEGREE = 3
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 0.01 (8/20 folds)
#Top gamma: 0.001 (10/20 folds)
#Top (C, gamma): (0.01, 0.01) (6/20 folds)
#Accuracy: 0.2845
#Balanced Accuracy: 0.2746

#Twelfth run (poly kernel grid: C, gamma, degree, coef0)
#GRID_C_VALUES = [0.1, 1, 10]
#GRID_GAMMA_VALUES = ["scale", 0.05, 0.1, 0.2]
#GRID_DEGREES = [2, 3]
#GRID_COEF0_VALUES = [0, 0.5, 1, 2]
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 0.1 (15/20 folds)
#Top gamma: 0.05 (13/20 folds)
#Top degree: 3 (11/20 folds)
#Top coef0: 2.0 (7/20 folds)
#Top (C, gamma, degree, coef0): (0.1, 0.05, 2, 2.0) (4/20 folds)
#Accuracy: 0.5061
#Balanced Accuracy: 0.5195

#Thirteenth run (poly kernel, fixed degree/coef0 with C/gamma sweep)
#FIXED_KERNEL = "poly"
#FIXED_DEGREE = 3
#FIXED_COEF0 = 2.0
#GRID_C_VALUES = [0.05, 0.1, 0.2, 0.5, 1]
#GRID_GAMMA_VALUES = ["scale", 0.02, 0.05, 0.1]
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 0.05 (13/20 folds)
#Top gamma: 0.02 (13/20 folds)
#Top (C, gamma): (0.05, 0.02) (9/20 folds)
#Accuracy: 0.5745
#Balanced Accuracy: 0.5827

#Fourteenth run (poly kernel, fixed degree=2/coef0 with C/gamma sweep)
#FIXED_KERNEL = "poly"
#FIXED_DEGREE = 2
#FIXED_COEF0 = 2.0
#GRID_C_VALUES = [0.05, 0.1, 0.2, 0.5, 1]
#GRID_GAMMA_VALUES = ["scale", 0.02, 0.05, 0.1]
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 0.2 (5/20 folds)
#Top gamma: 0.02 (13/20 folds)
#Top (C, gamma): (0.2, 0.02) (5/20 folds)
#Accuracy: 0.5782
#Balanced Accuracy: 0.5897

#Fifteenth run (poly kernel, degree=2/coef0 fixed, expanded low-gamma test)
#FIXED_KERNEL = "poly"
#FIXED_DEGREE = 2
#FIXED_COEF0 = 2.0
#GRID_C_VALUES = [0.1, 0.15, 0.2, 0.3, 0.5]
#GRID_GAMMA_VALUES = ["scale", 0.01, 0.02, 0.05, 0.001, 0.005]
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 0.5 (6/20 folds)
#Top gamma: 0.02 (8/20 folds)
#Top (C, gamma): (0.5, 0.01) (3/20 folds)
#Accuracy: 0.5489
#Balanced Accuracy: 0.5626

#Sixteenth run (rbf kernel, high-C sweep near 50)
#FIXED_KERNEL = "rbf"
#GRID_C_VALUES = [50, 75, 100]
#GRID_GAMMA_VALUES = ["scale", 0.01, 0.02, 0.05, 0.001, 0.005]
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 50.0 (11/20 folds)
#Top gamma: 0.001 (14/20 folds)
#Top (C, gamma): (50.0, 0.001) (11/20 folds)
#Accuracy: 0.5928
#Balanced Accuracy: 0.6014

#Seventeenth run (rbf kernel, local search around C~50 and gamma~0.001)
#FIXED_KERNEL = "rbf"
#GRID_C_VALUES = [35, 50, 65, 80]
#GRID_GAMMA_VALUES = [0.0005, 0.001, 0.002, 0.005]
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 35.0 (7/20 folds)
#Top gamma: 0.0005 (9/20 folds)
#Top (C, gamma): (35.0, 0.0005) (4/20 folds)
#Accuracy: 0.5871
#Balanced Accuracy: 0.5967

#Eighteenth run (rbf kernel, fixed C=40 with small gamma sweep)
#FIXED_KERNEL = "rbf"
#GRID_C_VALUES = [40]
#GRID_GAMMA_VALUES = ["scale", 0.001, 0.0005]
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 40.0 (20/20 folds)
#Top gamma: 0.001 (13/20 folds)
#Top (C, gamma): (40.0, 0.001) (13/20 folds)
#Accuracy: 0.5883
#Balanced Accuracy: 0.5989

#Nineteenth run (rbf kernel, gamma fixed at 0.001 with C sweep 20-50)
#FIXED_KERNEL = "rbf"
#GRID_C_VALUES = [20, 25, 30, 35, 40, 45, 50]
#GRID_GAMMA_VALUES = [0.001]
#FIXED_CLASS_WEIGHT = "balanced"
#Top C: 50.0 (6/20 folds)
#Top gamma: 0.001 (20/20 folds)
#Top (C, gamma): (50.0, 0.001) (6/20 folds)
#Accuracy: 0.5894
#Balanced Accuracy: 0.5996


