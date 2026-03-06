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

cwd = getcwd()
# project root directory
dspath = Path(cwd).parent.parent
dataset = "ds004504"

