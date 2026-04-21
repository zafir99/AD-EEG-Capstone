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
# Wide search to explore very different model complexity regimes.
GRID_C_VALUES = [0.01, 0.1, 1, 10, 100]
GRID_GAMMA_VALUES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
FIXED_CLASS_WEIGHT = "balanced"

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
    best_c_values: list[float] = []
    best_gamma_values: list[float] = []
    best_param_pairs: list[tuple[float, float]] = []

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
                ("svc", SVC(kernel="rbf", class_weight=FIXED_CLASS_WEIGHT)),
            ]
        )

        # Tune C and gamma with class_weight fixed to balanced.
        param_grid = {
            "svc__C": GRID_C_VALUES,
            "svc__gamma": GRID_GAMMA_VALUES,
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
            scoring="balanced_accuracy",
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
        best_c = float(search.best_params_["svc__C"])
        best_gamma = float(search.best_params_["svc__gamma"])
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

    c_counter = Counter(best_c_values)
    gamma_counter = Counter(best_gamma_values)
    pair_counter = Counter(best_param_pairs)
    top_c, top_c_count = c_counter.most_common(1)[0]
    top_gamma, top_gamma_count = gamma_counter.most_common(1)[0]
    top_pair, top_pair_count = pair_counter.most_common(1)[0]

    print("\n=== Best Parameter Frequency (Across 20 LOSO Folds) ===")
    print(f"C counts: {dict(sorted(c_counter.items()))}")
    print(f"Gamma counts: {dict(sorted(gamma_counter.items()))}")
    print(f"Top C: {top_c} ({top_c_count}/20 folds)")
    print(f"Top gamma: {top_gamma} ({top_gamma_count}/20 folds)")
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
#Might need to test some more?

