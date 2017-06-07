"""
This script evaluates a few classification methods on 'naively' computed features.
It loads the data from 'raw' (mne) format, applies FFT on each channel data, then computes the magnitude of each of
three bands + overall - average over each channel.
A feature vector then has a size of (num_channels + 1) x num_bands = 20 x 3.

This script will try to load the data if computed before.

SVM, RandomForest, and LogisticRegression are then used to fit the data.
For each method, there is a dictionary with it's parameters (see the 'Learning Representations from EEG...' Paper).
For each value (combination of values) of the parameters, accuracy of the classifier is computed using cross validation,
with k = 5.

The results are saved to a .csv file (results.csv).

Have fun!
"""

import os
import csv

import h5py
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from EEGLearn.raw_to_image import sample_to_channels

hdf_filepath = '/datasets/CogReplay/dl-eeg/extracted_raw.hdf5'
#hdf_filepath = './raw.hdf5'
h5_file = h5py.File(hdf_filepath, 'r')
bands_filepath = 'bands.npy'
if not os.path.exists(bands_filepath):
    print('Generating data.')
    bands_shape = (
        # num samples, num channels + 1 for avg over all, num bands
        h5_file['data'].shape[0], h5_file['data'].shape[1] + 1, 3,
    )
    bands_data = np.empty(bands_shape)
    for row_idx, data_label in tqdm(enumerate(zip(h5_file['data'], h5_file['labels'])), total=766):
        data, label = data_label
        for ch_id, ch_data in enumerate(data):
            # compute characteristics for a channel
            samples = []
            sample_rate = 1 / 128
            window_data = ch_data
            fft = np.fft.fft(window_data)
            freqs = np.fft.fftfreq(len(fft), sample_rate)
            theta, alpha, beta = sample_to_channels(
                sample=fft,
                freqs=freqs
            )
            bands_data[row_idx, ch_id, 0] = theta.__abs__()
            bands_data[row_idx, ch_id, 1] = alpha.__abs__()
            bands_data[row_idx, ch_id, 2] = beta.__abs__()

        for band_idx in range(3):
            bands_data[row_idx, 19, band_idx] = np.average(bands_data[row_idx, :, band_idx])

    print('Saving to data to {}'.format(bands_filepath))
    np.save(bands_filepath, bands_data)
else:
    print('Loading from data from {}'.format(bands_filepath))
    bands_data = np.load(bands_filepath)

bands_data = bands_data.reshape((bands_data.shape[0], bands_data.shape[1] * bands_data.shape[2]))
targets = np.array(h5_file['labels'])

results_filepath = 'results.csv'
with open(results_filepath, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # write header
    csv_writer.writerow(['Score', 'Score std', 'Method', 'Method Params'])

    print('Testing SVMs:')
    csv_writer.writerow(['', '', '', 'C', 'gamma'])
    svm_params = {
        'C': (0.01, 0.1, 1, 10, 10),
        'gamma': [i / 10 for i in range(1, 10)] + [i for i in range(1, 11)] + ['auto']
    }
    for C in svm_params['C']:
        for gamma in svm_params['gamma']:
            clf = svm.SVC(C=C, gamma=gamma)
            scores = cross_val_score(clf, bands_data, targets, cv=5)
            print('C: {}, gamma: {}'.format(C, gamma))
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            csv_writer.writerow([scores.mean(), scores.std() * 2, 'SVC', C, gamma])

    print('Testing Random Forests:')
    csv_writer.writerow(['', '', '', 'num estimators'])
    random_forest_params = {
        'n_estimators': (5, 10, 20, 50, 100, 500, 1000)
    }
    for n_estimators in random_forest_params['n_estimators']:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=None,
        )
        scores = cross_val_score(clf, bands_data, targets, cv=5)
        print('Num. estimators: {}'.format(n_estimators))
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        csv_writer.writerow([scores.mean(), scores.std() * 2, 'RandomForest', n_estimators])

    print('Testing Logistic Regression:')
    csv_writer.writerow(['', '', '', 'C'])
    logistic_regr_params = {
        'C': (0.01, 0.1, 1, 10, 100, 1000,)
    }
    for C in logistic_regr_params['C']:
        clf = LogisticRegression(C=C)
        scores = cross_val_score(clf, bands_data, targets, cv=5)
        print('C: {}'.format(C))
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        csv_writer.writerow([scores.mean(), scores.std() * 2, 'LogisticRegression', C])
