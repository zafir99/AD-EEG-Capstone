import openneuro
import numpy as np
from os import getcwd, makedirs
from pathlib import Path
from mne.io import Raw

from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)

def process_psd (bids_path : BIDSPath, num_subjects : int, freq_bands : Tuple[Tuple[int]],
                 num_channels : int, num_bands : int) -> np.ndarray :

    subject_avgs = np.zeros(shape=(num_subjects, num_bands, num_channels), dtype=np.float64)
    for i in range(num_subjects) :
        bids_path.update(subject=str(i+1).zfill(3))
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        spec = raw.compute_psd(method="welch", verbose=False)

        for j in range(num_bands) :
            # get power average for each channel along each frequency band
            mean_data = spec.get_data(fmin=freq_bands[j][0], fmax=freq_bands[j][1]).mean(axis=1)
            np.copyto(subject_avgs[i][j], mean_data)
            del mean_data

        # cleanup
        raw.close()
        del raw
    return subject_avgs


cwd = Path(getcwd())
# project root directory
dspath = cwd.parent.parent
dataset = "ds004504"

# bids file path
bids_root = dspath / dataset

# check if dataset exists, download if it doesn't
if not bids_root.exists() :
    openneuro.download(dataset=dataset, target_dir=bids_root)

# make output folder
out_folder = cwd.parent / "out"
makedirs(name=out_folder, mode=0o777, exist_ok=True)

# eeg data setup
datatype = "eeg"
extensions = [".set", ".tsv"]
task = "eyesclosed"

# subject ranges
alzheimers = (1,36)
control = (37,65)
ftd = (66,88)
# only interested in alzheimers and control patients
num_subjects = 65

# frequency bands
delta = (1,4)
theta = (4,8)
alpha = (8,13)
beta  = (13,30)
freq_bands = (delta, theta, alpha, beta)
num_bands = len(freq_bands)

# all channel types in ds004504
channels = ('Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz')
num_channels = len(channels)

# set bids base path
bids_path = BIDSPath(root=bids_root, task=task, suffix=datatype,
                     extension=extensions[0], datatype=datatype)

psd_channel_avgs = process_psd(bids_path=bids_path, num_subjects=num_subjects,
                               freq_bands=freq_bands, num_channels=num_channels,
                               num_bands=num_bands)

a_channel_avg = psd_channel_avgs[alzheimers[0]-1:alzheimers[1]].mean(axis=0)
c_channel_avg = psd_channel_avgs[control[0]-1:control[1]].mean(axis=0)

a_fname = out_folder / "alz_channel_band_avg.csv"
c_fname = out_folder / "con_channel_band_avg.csv"
af = open(a_fname, "wt")
cf = open(c_fname, "wt")
af.write("Channel,Delta_Power,Theta_Power,Alpha_Power,Beta_Power\n")
cf.write("Channel,Delta_Power,Theta_Power,Alpha_Power,Beta_Power\n")

for i in range (num_channels) :
    af.write(channels[i] + ',' + str(a_channel_avg[0][i]) + ',' + str(a_channel_avg[1][i]) + ',' +
             str(a_channel_avg[2][i]) + ',' + str(a_channel_avg[3][i]) + '\n')

    cf.write(channels[i] + ',' + str(c_channel_avg[0][i]) + ',' + str(c_channel_avg[1][i]) + ',' +
             str(c_channel_avg[2][i]) + ',' + str(c_channel_avg[3][i]) + '\n')

af.close()
cf.close()
