## Libraries
* openneuro-py (optional, but recommended)
    * **Not necessary for downloading dataset, but would be helpful, more details below**
    * Documentation: [openneuro-py](https://github.com/openneuro-py/openneuro-py)
* mne-bids
    * Documentation: [mne-tools](https://mne.tools/mne-bids/stable/api.html)
* mne
    * Documentation: [mne](https://github.com/mne-tools/mne-python)

## Dataset Downloading
**The MNE BIDS library has to install a sample set of files for its initial setup (~1.5GB). In [`vis.py`](https://github.com/zafir99/AD-EEG-Capstone/blob/phase_1/vis.py), the sample files are set up to be installed in the current directory.**
1. Download with openneuro cli (recommended)
     * `openneuro-py download --dataset=ds004504 --target-dir=`whatever directory you want
     * If the `--target-dir` option is not provided, the current directory will be used by default. Keep in mind the sample library is already being downloaded in the current directory, so this would be most convenient given the current setup.
2. Download with aws
     * `aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ds004504-download/`
3. Download with datalad
     * `datalad install https://github.com/OpenNeuroDatasets/ds004504.git`
