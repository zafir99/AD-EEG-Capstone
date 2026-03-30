import openneuro
import numpy as np
from os import getcwd, makedirs
from pathlib import Path
from mne.io import Raw
from mne import make_fixed_length_epochs as mfl_epochs

from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)

# maybe im an idiot for writing this method
# God knows best
def write_csv (fname : str, len : int, data : np.ndarray, root_dir : Path) :
    real_path = root_dir / fname
    f = open(real_path, "wt")
    f.write("Delta_Power,Theta_Power,Alpha_Power,Beta_Power\n")

    for i in len :
        f.write(str(data[i][0]) + ',' + str(data[i][1]) + ',' +
                str(data[i][2]) + ',' + str(data[i][3]) + '\n')

    print(f"\"{fname}\" successfully made!")
    f.close()
    return


def process_psd (bids_path : BIDSPath, num_subjects : int, freq_bands : Tuple[Tuple[int]],
                 num_channels : int, num_bands : int) -> np.ndarray :

    subject_data = list()
    epoch_dur = 4.0
    overlap_ratio = 0.5
    overlap = overlap_ratio*epoch_dur

    for i in range(num_subjects) :
        bids_path.update(subject=str(i+1).zfill(3))
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        epochs = mfl_epochs(raw=raw, duration=epoch_dur, overlap=overlap)
        specs = list(map(compute_psd(method="welch", verbose=False), epochs))

        for j in range(num_bands) :
            d

        # cleanup
        raw.close()
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

# make output folder
out_folder = cwd.parent / "processed"
makedirs(name=out_folder, mode=0o777, exist_ok=True)

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

# frequency bands
delta = (0.5,4)
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

a_data = psd_channel_avgs[alzheimers[0]-1:alzheimers[1]].mean(axis=2)
c_data = psd_channel_avgs[control[0]-1:control[1]].mean(axis=2)

a_fname = "alz_epoch_rbp.csv"
c_fname = "con_epoch_rbp.csv"
f_fname = "ftd_epoch_rbp.csv"

write_csv(filename=a_fname, root_dir=out_folder, data=a_channel_avg, len=num_a)
write_csv(filename=c_fname, root_dir=out_folder, data=c_channel_avg, len=num_c)
write_csv(filename=f_fname, root_dir=out_folder, data=f_channel_avg, len=num_f)

