## Acknowledgements
**We aknowledge support of the 2nd Department of Neurology of AHEPA General University Hospital of Thessaloniki. We acknowledge support of this work from the project “Immersive Virtual, Augmented and Mixed Reality Center of Epirus” (MIS 5047221) which is implemented under the Action “Reinforcement of the Research and Innovation Infrastructure”, funded by the Operational Programme “Competitiveness, Entrepreneurship and Innovation” (NSRF 2014-2020) and co-financed by Greece and the European Union (European Regional Development Fund).**

**Andreas Miltiadous, Katerina D. Tzimourta, Theodora Afrantou, Panagiotis Ioannidis, Nikolaos Grigoriadis, Dimitrios G. Tsalikakis, Pantelis Angelidis, Markos G. Tsipouras, Evripidis Glavas, Nikolaos Giannakeas, and Alexandros T. Tzallas (2024). A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds004504.v1.0.7**

**Data descriptor: 10.3390/data8060095 First study on this dataset: 10.1109/ACCESS.2023.3294618**

## Libraries
* **[openneuro-py](https://github.com/openneuro-py/openneuro-py)**
* **[mne-bids](https://mne.tools/mne-bids/stable/api.html)**
* **[mne](https://github.com/mne-tools/mne-python)**
* **[scipy](https://docs.scipy.org/doc/scipy/)**
* **[scikit-learn](https://scikit-learn.org/stable/user_guide.html)**

## Dataset Downloading
**In [`svg.py`](https://github.com/zafir99/AD-EEG-Capstone/blob/svm/src/svm.py), the dataset will automatically be downloaded in the project root directory, and operations on the dataset assume this directory structure. If you would like to manually install the dataset, instructions below are provided.**
1. Download with openneuro cli (recommended)
     * `openneuro-py download --dataset=ds004504 --target-dir=<whatever_directory>`
     * If the `--target-dir` option is not provided, the current directory will be used by default. Keep in mind that it is assumed the dataset exists in the project root directory, so downloading it there would be most convenient.
2. Download with aws
     * `aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ds004504-download/`
3. Download with datalad
     * `datalad install https://github.com/OpenNeuroDatasets/ds004504.git`

