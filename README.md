## Libraries
* **[openneuro-py](https://github.com/openneuro-py/openneuro-py)**
* **[mne-bids](https://mne.tools/mne-bids/stable/api.html)**
* **[mne](https://github.com/mne-tools/mne-python)**

## Dataset Downloading
**In [`vis.py`](https://github.com/zafir99/AD-EEG-Capstone/blob/phase_1/stage1/src/vis.py), the dataset will automatically be downloaded in the project root directory, and operations on the dataset assume this directory structure. If you would like to manually install the dataset, instructions below are provided.**
1. Download with openneuro cli (recommended)
     * `openneuro-py download --dataset=ds004504 --target-dir=<whatever_directory>`
     * If the `--target-dir` option is not provided, the current directory will be used by default. Keep in mind that it is assumed it exists in the project root directory, so downloading it there would be most convenient.
2. Download with aws
     * `aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ds004504-download/`
3. Download with datalad
     * `datalad install https://github.com/OpenNeuroDatasets/ds004504.git`

## Potential Datasets for Exploration
**There are datasets with valuable EEG data of Alzheimer's patients and healthy individuals but are not formatted in the BIDS standard. In the interest of time, all exploratory datasets below are from OpenNeuro, where the BIDS file format is guaranteed.**
* **[ds006036](https://openneuro.org/datasets/ds006036/versions/1.0.5)**
* **[ds005048](https://openneuro.org/datasets/ds005048/versions/1.0.1)**
