import argparse
import os
import re

import h5py as h5py
import sys
from mne.io import *
# progress bar
from tqdm import tqdm
import numpy as np

# channels with these indexes will be skipped
# TODO: make sure (ask Sebastian) if we can really skip them
ARTIFICIAL_CHANNELS = (5, 11, 18, 22,)


def find_files(folder, extension='.mat'):
    """
    Finds files with a given extension, looking in the given folder and subdirectories (recursively).
    :param folder: path to the folder to look
    :param extension: str, eg.: '.mat'
    :return: list of file paths to found files
    """
    filepaths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                filepaths.append(os.path.join(root, file))
    return filepaths


def extract_raw(filepaths):
    """
    Extracts data in matlab's EEGLab format using the MNE library.
    Also saves the maximal encountered value among all the data (which can be used for normalization),
    and the length of the longest series.
    :param filepaths: list of file paths to the files to be extracted.
    :return:
        subjects - list of (integers) labels for each file
        subjects_data - list of rows (numpy arrays) of data for a file
        locs - list of lists of channel locations
        max_series_len - longest sequence found in the data (used for padding)
        max_val - (float) maximal value found among channel data
    """
    max_series_len = 0
    max_val = sys.float_info.min
    subjects, subjects_data, locs = [], [], []
    for filepath in tqdm(filepaths, desc='Extracting'):
        # parse the subject index
        subject = re.search('S\d{2}', os.path.basename(filepath)).group(0)
        subject = int(subject.replace('S', ''))

        raw = read_raw_eeglab(filepath, verbose='CRITICAL')
        data, times = raw[:, :]
        # used for padding later
        max_series_len = max(max_series_len, len(times))

        channels_data, channels_locs = [], []
        for channel_idx, channel_data in enumerate(data):
            # skip channels from ARTIFICIAL_CHANNELS
            if channel_idx in ARTIFICIAL_CHANNELS:
                continue

            # this will be used for normalizing
            max_val = max(max_val, channel_data.max())

            # extract channel location
            channel_loc = raw.info['chs'][channel_idx]['loc'][:3]
            channels_data.append(channel_data)
            channels_locs.append(channel_loc)

        # append data for a file
        subjects.append(subject)
        subjects_data.append(np.array(channels_data))
        locs.append(channels_locs)

    return subjects, subjects_data, locs, max_series_len, max_val


def process_data(data, max_series_len=None, max_val=None):
    """
    Normalizes and pads the data.
    :param data: list of rows of data to be processed.
    :param max_series_len: integer.
        Maximal length of a series of data. All rows of data (and for each channel) will be
        padded with 0 to this length.
    :param max_val: float.
        All values in the data will be divided by this.
    :return: processed data
    """
    if max_val is not None:
        for subject_data in tqdm(data, desc='Normalizing'):
            subject_data /= max_val
    if max_series_len is not None:
        data = [np.pad(subject_data, ((0, 0), (0, max_series_len - subject_data.shape[1])), 'constant') for
                subject_data in tqdm(data, desc='Padding')]

    return data


def save_to_h5(h5_filepath, labels, locs, data):
    """
    Saves the data to a .hdf5 file with the necessary attributes - total number of unique labels,
    and dimensionality of the data (num_channels x max_series_len).
    :param h5_filepath: path to the output file
    :param labels: list of integers indicating labels for each row of data
    :param locs: list of lists of 3-dimensional locations of channels
    :param data: list of rows of data, where each row should have dimensionality: num_channels x max_series_len
    """
    hdf5_file = h5py.File(h5_filepath, 'w')
    hdf5_data = hdf5_file.create_dataset('data', data=data)
    hdf5_labels = hdf5_file.create_dataset('labels', data=labels)
    hdf5_file.create_dataset('locs', data=locs)

    hdf5_data.attrs['dims'] = data[0].shape
    hdf5_labels.attrs['num_labels'] = len(set(labels))
    hdf5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data extraction')
    parser.add_argument('-d', '--data_folder', default='./data.import')
    parser.add_argument('-t', '--target', default='extracted.hdf5')
    parser.add_argument('-i', '--images', action='store_true')
    args = parser.parse_args()

    files = find_files(args.data_folder)
    print('Files found: {}'.format(len(files)))
    subjects, subjects_data, locs, max_series_len, max_val = extract_raw(filepaths=files[:50])
    subjects_data = process_data(
        data=subjects_data,
        max_series_len=max_series_len,
        max_val=max_val,
    )
    if args.images:
        print('Converting to images')
        # TODO

    print('Saving to h5')
    save_to_h5(
        h5_filepath=args.target,
        labels=subjects,
        locs=locs,
        data=subjects_data,
    )
