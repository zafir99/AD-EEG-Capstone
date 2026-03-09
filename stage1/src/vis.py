import openneuro
from os import getcwd, makedirs
from pathlib import Path
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)

def read_and_plot (bids_path : BIDSPath, subject : str, suffix : str, task : str,
                   filename : str, channel : str) :

    bids_path = bids_path.update(subject=subject, suffix=suffix, task=task)
    raw = read_raw_bids(bids_path, verbose=False).pick(picks=channel)
    # DEBUG MESSAGE
    # raw.describe(False)

    # setting plotting time to 10 seconds
    dur=10
    # basic plot formatting, show=False means no window popup, clipping=1 means line can go out of bounds
    fig = raw.plot(time_format="float", show_scrollbars=False, show=False, duration=dur, clipping=1)
    title = f"{channel} Channel Readings for Subject {subject} ({filename})"

    # further plot formatting
    fig.suptitle(t=title, y = 0.965, size="xx-large", weight="demi")
    fig.subplots_adjust(left=-0.02, top=0.92, bottom=0.03)
    fig.set_dpi(200)
    fig.savefig(filename)
    raw.close()
    return


dataset = "ds004504"
cwd = Path(getcwd())

# project root directory
dspath = cwd.parent.parent
# set bids root directory
bids_root = dspath / dataset

# downloading dataset to bids root directory only if it does not exist
if not bids_root.exists() :
    openneuro.download(dataset=dataset, target_dir=bids_root)

# DEBUG MESSAGES TO CONSOLE
# print_dir_tree(bids_root, max_depth=4)
# print(make_report(bids_root))

# ds004504 contains EEG waveforms in the .set files, .tsv files are for formatting
# ignore .json files
extensions = [".set", ".tsv"]
datatype = "eeg"
task = "eyesclosed"

# keep plots in directory above source code
img_folder = cwd.parent / "plots"
alzheimers = ["001", img_folder / "Alzheimers"]
control = ["037",  img_folder / "Control"]
channel = "F4"

# set bids file format path
bids_path = BIDSPath(root=bids_root, datatype=datatype)
# make image plot directory
makedirs(name=img_folder, mode=0o777, exist_ok=True)
# read in data and plot
read_and_plot(bids_path=bids_path, subject=alzheimers[0], suffix=datatype,
              task=task, filename=alzheimers[1], channel=channel)
read_and_plot(bids_path=bids_path, subject=control[0], suffix=datatype,
              task=task, filename=control[1], channel=channel)
