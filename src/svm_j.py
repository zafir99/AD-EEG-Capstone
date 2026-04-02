"""
File: svm.py
Author: Juan Pablo Laseter Jr., whoever esle edits this mess
Date: 2026-03-31
Description: SVM model that reads the inputs of subjects and classifies
whether a person has alzheimers or not.
"""

# Imports
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# Config
# Pre-dowlad the dataset, unsure about how to do it in this file lol
BASE_PATH = "./ds004504"   # dataset folder, must be in the same folder as this file
FS = 256                   # sampling frequency

# Feature Extraction---------------------------------------------------------
def extract_band_powers(signal, fs):
    signal = np.asarray(signal).flatten()

    # Skip bad/short signals
    if len(signal) < fs:
        return [0, 0, 0, 0]

    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))

    def band_power(fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapz(psd[idx], freqs[idx])

    return [
        band_power(0.5, 4),   # Delta
        band_power(4, 8),     # Theta
        band_power(8, 13),    # Alpha
        band_power(13, 30)    # Beta
    ]

# Load Dataset---------------------------------------------------------------
def load_dataset():
    X = [] # Store Featured
    y = [] # Store labels

    print("Scanning dataset...\n")

    # Recursively search through subfolders for eeg files
    for root, dirs, files in os.walk(BASE_PATH):
        for file in files: # Loop through each file in current directory
            if file.endswith(".set"):
                file_path = os.path.join(root, file)
                print("Loading:", file_path)

                # Attempt to read the contents of the file
                try:
                    raw = mne.io.read_raw_eeglab(file_path, preload=True)

                    # Get sampling frequency dynamically
                    fs = int(raw.info['sfreq'])

                    # Only eeg channels
                    data = raw.get_data(picks="eeg")

                    print("Data shape:", data.shape) # Show the # of channels and time points in data

                    sample_features = [] # Store the eeg features for this one subj in arr

                    for channel in data:
                        feats = extract_band_powers(channel, fs) # compute a, b, t, d powers
                        sample_features.extend(feats) # pass list, convert into a giant vector of channel features

                    # Skip empty samples
                    if len(sample_features) == 0:
                        print("Skipping empty sample")
                        continue

                    X.append(sample_features)

                    # LABELING (REPLACE THIS)
                    # TEMP: split dataset in half
                    # Need to read the participants .tsv and associate them with the proper subj
                    subject_id = root.split("sub-")[-1][:3]

                    if subject_id.isdigit():
                        if int(subject_id) > 36:
                            y.append(0)  # Control
                        else:
                            y.append(1)  # Alzheimer
                    else:
                        y.append(0)

                except Exception as e:
                    print("Error loading file:", e)

    return np.array(X), np.array(y)

# Train + Evaluate
def train_svm(X, y):
    print("\nTraining SVM...\n")

    # Split the ds into .8 training, .2 testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Comp standard deviation from training data
    X_test = scaler.transform(X_test) # Apply scaling to test data

    # Kernel = rbf, use radial basis function, handles non linear patterns
    # C = 1.0, High c-> fit tightly possibel overfitting, Low c-> smoother boundry better gen
    # Gamma = scale, controll how far inluence of each pint reaches
    # Class_weight = balanced, adjust imbalences, give importance to minority
    model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced') # define the classifier
    model.fit(X_train, y_train) # learn patterns from data

    y_pred = model.predict(X_test) # use trained model to predict labels for test

    # percision = how many correctly predicted
    # recall = how many actual positives found
    # f1-score = balance between percision and recall
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model

def plot_svm_classification()

# Main
if __name__ == "__main__":
    X, y = load_dataset()

    print("\nFinal Dataset Shape:", X.shape)
    print("Labels Shape:", y.shape)

    if len(X) == 0:
        print("[X] No data loaded. Check paths.")
    else:
        model = train_svm(X, y)
        
        #ts not working fuck my chud life
#        fig, ax = plt.subplots(figsize=(8, 6))
#        DecisionBoundryDisplay.from_estimator(
#            model,
#            X,
#            cmap=plt.cm.coolwarm,
#            response_method="predict",
#            ax=ax,
#            alpha=0.6,
#        )
#        
#
#        #Create scatterplot vis
#        probs = model.predict(X)
#        alz_probs = probs[:, 1] # Probability of alzheimers
#        pca = PCA(n_components = 2)
#        X_2d = pca.fit_transform(X)
#        
#        plt.figure()
#
#        plt.scatter(
#            X_2d[:, 0],
#            X_2d[:, 1],
#            c=(y_pred == y),  # correct vs incorrect
#        )
#
#        plt.xlabel("PCA Component 1")
#        plt.ylabel("PCA Component 2")
#        plt.title("Correct vs Incorrect Predictions")
#        plt.show()
