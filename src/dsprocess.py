#!/usr/bin/python3

import openneuro
import numpy as np
from os import getcwd, makedirs
from pathlib import Path
from mne.io import Raw
from mne import Epochs, make_fixed_length_epochs as mfl_epochs
from scipy.integrate import simpson

from mne_bids import (
    BIDSPath,
    print_dir_tree,
    read_raw_bids,
)

# maybe im an idiot for writing this method
# God knows best
def write_rbp_to_csv (sub_len : int,
                      offset : int,
                      data : list[np.ndarray],
                      root_dir : Path) :

    for i in range(sub_len) :
        fname = "sub_" + str(i+offset).zfill(3) + "_rbp.csv"
        real_path = root_dir / fname
        f = open(real_path, "wt")
        f.write("Delta_Power,Theta_Power,Alpha_Power,Beta_Power,Gamma_Power\n")
        num_epochs = data[i].shape[0]
        for j in range(num_epochs) :
            f.write(str(data[i][j][0]) + ',' + str(data[i][j][1]) + ',' +
                    str(data[i][j][2]) + ',' + str(data[i][j][3]) + ',' + 
                    str(data[i][j][4]) + '\n')
        print(f"\"{fname}\" successfully made!")
        f.close()
    return


# compute relative band power for every epoch of every subject
def process_rbp (bids_path : BIDSPath,
                 num_subjects : int,
                 freq_bands : Tuple[Tuple[int]],
                 num_bands : int) -> list[np.ndarray] :

    subject_data = []
    epoch_dur = 4.0
    overlap_ratio = 0.5
    overlap = overlap_ratio*epoch_dur
    fmin = freq_bands[0][0]
    fmax = freq_bands[num_bands-1][1]

    for i in range(num_subjects) :
        bids_path.update(subject=str(i+1).zfill(3))
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        epochs = mfl_epochs(raw=raw, duration=epoch_dur, overlap=overlap)
        # dont need to drop any epochs
        # load data from disk
        epochs.load_data()
        e_len = len(epochs)

        specs = epochs.compute_psd(method="welch", verbose=False, fmin=fmin, fmax=fmax, remove_dc=False)
        total_psd = simpson(y=np.absolute(specs.get_data()), axis=-1).sum(axis=1)
        arr = np.zeros(shape=(e_len,num_bands), dtype=np.float64)

        for j in range(num_bands) :
            freq_data = simpson(y=np.absolute(
                                specs.get_data(fmin=freq_bands[j][0],
                                               fmax=freq_bands[j][1])
                                ),
                                axis=-1).sum(axis=1)
            for k in range (e_len) :
                rbp = freq_data[k] / total_psd[k]
                arr[k][j] = rbp

        # append data to overall list
        subject_data.append(arr)
        # cleanup
        raw.close()
        del specs
        del epochs
        del raw
    return subject_data


cwd = Path(getcwd())
# project root directory
dspath = cwd.parent
dataset = "ds004504"

# bids file path
bids_root = dspath / dataset / "derivatives"

# check if dataset exists, download if it doesn't
if not bids_root.exists() :
    openneuro.download(dataset=dataset, target_dir=bids_root)

# eeg data setup
datatype = "eeg"
extensions = [".set", ".tsv"]
task = "eyesclosed"

# subject ranges
alzheimers = (1,36)
control = (37,65)
ftd = (66,88)

num_a = alzheimers[1]-alzheimers[0]+1
num_c = control[1]-control[0]+1
num_f = ftd[1]-ftd[0]+1
num_subjects = 88

# frequency bands
delta = (0.5,4)
theta = (4,8)
alpha = (8,13)
beta  = (13,30)
gamma = (30,45)
freq_bands = (delta, theta, alpha, beta, gamma)
num_bands = len(freq_bands)

# all channel types in ds004504
channels = ('Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz')
num_channels = len(channels)

# set bids base path
bids_path = BIDSPath(root=bids_root, task=task, suffix=datatype,
                     extension=extensions[0], datatype=datatype)

rbp_data = process_rbp(bids_path=bids_path, num_subjects=num_subjects,
                       freq_bands=freq_bands, num_bands=num_bands)

# partition data
a_data = rbp_data[alzheimers[0]-1:alzheimers[1]]
c_data = rbp_data[control[0]-1:control[1]]
f_data = rbp_data[ftd[0]-1:ftd[1]]

# set out folders
out_folder = cwd.parent / "processed"
a_folder = out_folder / "alz"
c_folder = out_folder / "con"
f_folder = out_folder / "ftd"
makedirs(name=out_folder, mode=0o777, exist_ok=True)
makedirs(name=a_folder, mode=0o777, exist_ok=True)
makedirs(name=c_folder, mode=0o777, exist_ok=True)
makedirs(name=f_folder, mode=0o777, exist_ok=True)

write_rbp_to_csv(root_dir=a_folder, data=a_data, sub_len=num_a, offset=alzheimers[0])
write_rbp_to_csv(root_dir=c_folder, data=c_data, sub_len=num_c, offset=control[0])
write_rbp_to_csv(root_dir=f_folder, data=f_data, sub_len=num_f, offset=ftd[0])
