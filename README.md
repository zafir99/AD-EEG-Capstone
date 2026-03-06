## Libraries
* openneuro-py (optional, but recommended)
    * Documentation: [openneuro-py](https://github.com/openneuro-py/openneuro-py)
* mne-bids
    * Documentation: [mne-tools](https://mne.tools/mne-bids/stable/api.html)
* mne
    * Documentation: [mne](https://github.com/mne-tools/mne-python)

## Dataset Downloading
**The MNE BIDS library has to install a sample set of files for its initial setup (~1.5GB). In [`vis.py`](https://github.com/zafir99/AD-EEG-Capstone/blob/phase_1/stage1/src/vis.py), the sample files are set up to be installed in the project root directory. This is separate from the EEG dataset that is instructed to be downloaded.**
1. Download with openneuro cli (recommended)
     * `openneuro-py download --dataset=ds004504 --target-dir=<whatever_directory>`
     * If the `--target-dir` option is not provided, the current directory will be used by default. Keep in mind the sample library is being downloaded in the project root directory, so downloading it there would be most convenient.
2. Download with aws
     * `aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ds004504-download/`
3. Download with datalad
     * `datalad install https://github.com/OpenNeuroDatasets/ds004504.git`

## Potential Datasets for Exploration
**There are datasets with valuable EEG data of Alzheimer's patients and healthy individuals but are not formatted in the BIDS standard. In the interest of time, all exploratory datasets below are from OpenNeuro, where the BIDS file format is guaranteed.**
* **[ds006036](https://openneuro.org/datasets/ds006036/versions/1.0.5)**
* **[ds005048](https://openneuro.org/datasets/ds005048/versions/1.0.1)**
