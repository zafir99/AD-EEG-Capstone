#!/usr/bin/python3
from sklearn import svm
import pandas as pd
import numpy as np
from os import getcwd, makedirs, execv
from pathlib import Path


# set working directories
cwd = Path(getcwd())
data_root = cwd.parent / "processed"
alz_root = data_root / "alz"
con_root = data_root / "con"
ftd_root = data_root / "ftd"

# make processed dataset if it does not exist
if not (alz_root.exists() and con_root.exists() and ftd_root.exists()) :
    print("Processed dataset does not exist.\nNow running dsprocess.py...\n")
    execv("./dsprocess.py", [" "])

# subject ranges
alz_index = (1,36)
con_index = (37,65)
ftd_index = (66,88)

num_a = alz_index[1]-alz_index[0]+1
num_c = con_index[1]-con_index[0]+1
num_f = ftd_index[1]-ftd_index[0]+1
num_subjects = 88

# set band ranges
delta = (0.5,4)
theta = (4,8)
alpha = (8,13)
beta  = (13,25)
gamma = (25,45)
freq_bands = (delta, theta, alpha, beta, gamma)
num_bands = len(freq_bands)


# variable declaration
skip = 0
skip_root = alz_root
acc = 0
total_rec = 0

# loso testing for loop
for n in range(con_index[1]) :
    skip = n + alz_index[0]
    skip_file = "sub_" + str(skip).zfill(3) + "_rbp.csv"

    # set up directory for testing subject file
    if (skip >= con_index[0]) :
        skip_root = con_root

    # read in all alzheimers patient data and create an aggregate list of all
    alz_list = []
    for i in range (num_a) :
        idx = i+alz_index[0]
        if (idx == skip) :
            continue
        sub_file = "sub_" + str(idx).zfill(3) + "_rbp.csv"
        csv = pd.read_csv(alz_root / sub_file, sep=',')
        alz_list.append(csv)

    # create aggregate pandas dataframe, then convert to numpy array for svm training
    alz_data = pd.concat(alz_list).to_numpy(dtype=np.float64)
    del alz_list

    # repeat process with control patients
    con_list = []
    for i in range (num_c) :
        idx = i + con_index[0]
        if (idx == skip) :
            continue
        sub_file = "sub_" + str(idx).zfill(3) + "_rbp.csv"
        csv = pd.read_csv(con_root / sub_file, sep=',')
        con_list.append(csv)

    test = pd.read_csv(skip_root / skip_file, sep=',').to_numpy(dtype=np.float64)

    con_data = pd.concat(con_list).to_numpy(dtype=np.float64)
    del con_list

    # get number of alzheimers and control patients
    count_a_recs = alz_data.shape[0]
    count_c_recs = con_data.shape[0]
    # aggregate all patient data into a single numpy array
    total_data = np.concat([con_data, alz_data], axis=0)

    del alz_data
    del con_data

    # label count and order
    labels = ["control"] * count_c_recs + ["alzheimers"] * count_a_recs
    bin_labels = [0] * count_c_recs + [1] * count_a_recs

    # test subject label is assigned 1 if they have alzheimers
    # 0 if they dont
    expected = 1
    if (skip >= con_index[0]) :
        expected = 0

    # svm model setup
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    clf.fit(total_data,bin_labels)

    print(f"Testing subject {skip}'s data...")
    lab_arr = clf.predict(test)

    # get total number of records from a single testing instance
    total_rec += len(lab_arr)
    # count number of correct guesses by svm model
    acc += lab_arr.tolist().count(expected)

# get accuracy rate
acc /= total_rec
print(f"Accuracy Rate: {acc*100}%")
