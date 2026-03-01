from os import getcwd
from mne.datasets import sample
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
    # raw.describe(False)
    dur=10
    fig = raw.plot(time_format="float", show_scrollbars=False, show=False, duration=dur, clipping=1)
    title = f"{channel} Channel Readings for Subject {subject} ({filename})"
    fig.suptitle(t=title, y = 0.965, size="xx-large", weight="demi")
    fig.subplots_adjust(left=-0.02, top=0.92, bottom=0.03)
    fig.set_dpi(200)
    fig.savefig(filename)
    raw.close()
    return


dataset = "ds004504"

dspath = getcwd()
sample.data_path(path=dspath)
bids_root = sample.data_path().parent / dataset

# MESSAGES TO CONSOLE
# print_dir_tree(bids_root, max_depth=4)
# print(make_report(bids_root))

extensions = [".set", ".tsv"]
datatype = "eeg"
task = "eyesclosed"
alzheimers = ["001", "Alzheimers"]
control = ["037", "Control"]
channel = "F4"

bids_path = BIDSPath(root=bids_root, datatype=datatype)
read_and_plot(bids_path, alzheimers[0], datatype, task, alzheimers[1], channel)
read_and_plot(bids_path, control[0], datatype, task, control[1], channel)
