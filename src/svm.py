#!/usr/bin/python3
from sklearn import svm
import pandas as pd
import numpy as np
from os import getcwd, makedirs, execv
from pathlib import Path


cwd = Path(getcwd())
data_root = cwd.parent / "processed"
alz_root = data_root / "alz"
con_root = data_root / "con"
ftd_root = data_root / "ftd"

if not (alz_root.exists() and con_root.exists() and ftd_root.exists()) :
    print("Processed dataset does not exist.\nNow running dsprocess.py...\n")
    execv("./dsprocess.py", [" "])

alz_index = (1,36)
con_index = (37,65)
ftd_index = (66,88)

num_a = alz_index[1]-alz_index[0]+1
num_c = con_index[1]-con_index[0]+1
num_f = ftd_index[1]-ftd_index[0]+1
num_subjects = 88

delta = (0.5,4)
theta = (4,8)
alpha = (8,13)
beta  = (13,25)
gamma = (25,45)
freq_bands = (delta, theta, alpha, beta, gamma)
num_bands = len(freq_bands)


skip = 0
skip_root = alz_root
acc = 0
total_rec = 0

for n in range(con_index[1]) :
    skip = n + alz_index[0]
    if (skip >= con_index[0]) :
        skip_root = con_root

    alz_list = []
    for i in range (num_a) :
        idx = i+alz_index[0]
        if (idx == skip) :
            continue
        sub_file = "sub_" + str(idx).zfill(3) + "_rbp.csv"
        csv = pd.read_csv(alz_root / sub_file, sep=',')
        alz_list.append(csv)

    alz_data = pd.concat(alz_list).to_numpy(dtype=np.float64)
    del alz_list

    con_list = []
    for i in range (num_c) :
        idx = i + con_index[0]
        if (idx == skip) :
            continue
        sub_file = "sub_" + str(idx).zfill(3) + "_rbp.csv"
        csv = pd.read_csv(con_root / ("sub_" + str(idx).zfill(3) + "_rbp.csv"), sep=',')
        con_list.append(csv)

    test = pd.read_csv(skip_root / ("sub_" + str(skip).zfill(3) + "_rbp.csv"), sep=',').to_numpy(dtype=np.float64)

    con_data = pd.concat(con_list).to_numpy(dtype=np.float64)
    del con_list

    count_a_recs = alz_data.shape[0]
    count_c_recs = con_data.shape[0]
    total_data = np.concat([con_data, alz_data], axis=0)

    del alz_data
    del con_data

    labels = ["control"] * count_c_recs + ["alzheimers"] * count_a_recs
    bin_labels = [0] * count_c_recs + [1] * count_a_recs
    expected = 0
    if (skip <= alz_index[1]) :
        expected = 1

    clf = svm.SVC(kernel='poly', degree=5)
    clf.fit(total_data,bin_labels)

    print(f"Testing subject {skip}'s data...")
    lab_arr = clf.predict(test)

    total_rec += len(lab_arr)
    acc += lab_arr.tolist().count(expected)

acc /= total_rec
print(f"Accuracy Rate: {acc*100}%")
