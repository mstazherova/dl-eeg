import argparse
import re

import h5py as h5py
from mne.io import *
# progress bar
from openpyxl import load_workbook
from tqdm import tqdm
import numpy as np

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from EEGLearn.raw_to_image import raw_to_image

# channels with these indexes will be skipped
# TODO: make sure (ask Sebastian) if we can really skip them
ARTIFICIAL_CHANNELS = (5, 11, 18, 22,)


def find_files(path, extension='.mat'):
    """
    Finds files with a given extension, looking in the given folder and subdirectories (recursively).
    If path points to a file rather than a directory only this one file will be returned.
    :param path: path to the folder to look or a single .mat file
    :param extension: str, eg.: '.mat'
    :return: list of file paths to found files
    """

    # if the path specified is a single file
    if os.path.isfile(path):
        if path.endswith(extension):
            return [path]
        raise ValueError('If a single file is specified it must be a .mat file. \nGiven: {}'.format(path))

    filepaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                filepaths.append(os.path.join(root, file))
    return filepaths


def get_task_type(filepath):
    def get_task_type_from_xls(filepath):
        wb = load_workbook(filename='/datasets/CogReplay/data.import/CR_SegmentTracking_AllData_Sept1.xlsx')

        filepath = os.path.basename(filepath)
        segment_name = filepath.split('_MR_')[0]
        segment_name = '{}_MR'.format(segment_name)
        beg, end = filepath.split('_MR_')[1].split('.mat')[0].split('_')
        beg, end = int(beg), int(end)

        segment_flag = False
        for row in wb.worksheets[1].rows:
            if row[0].value and row[0].value == segment_name:
                segment_flag = True
            elif row[0].value:
                if segment_flag:
                    print('Havent found corresponding row for {}'.format(filepath))
                    return ''

            if segment_flag and row[2].value == beg:
                task_type = row[4].value
                return task_type

        print('Havent found corresponding row for {}'.format(filepath))
        return ''

    dream_type = '_DR'
    if '_MR_' in filepath:
        dream_type = '_MR'
        task_type = get_task_type_from_xls(filepath)
    else:
        task_type = os.path.basename(filepath).split(dream_type)[0].split('_')[-1]
    return '{}{}'.format(task_type, dream_type)


def extract_raw(filepaths, cut_from=0):
    """
    Extracts data in matlab's EEGLab format using the MNE library.
    Also saves the maximal encountered value among all the data (which can be used for normalization),
    and the length of the longest series.
    :param filepaths: list of file paths to the files to be extracted.
    :param cut_from: index (of points in time series) from which to start extracting the data
    :return:
        subjects - list of (integers) labels for each file
        subjects_data - list of rows (numpy arrays) of data for a file
        locs - list of lists of channel locations
        max_series_len - longest sequence found in the data (used for padding)
        max_val - (float) maximal value found among channel data
        sfreq - (int) sampling frequency of the data
    """
    max_series_len = 0
    max_val = sys.float_info.min
    subjects, subjects_data, locs, task_types = [], [], [], []
    for filepath in tqdm(filepaths, desc='Extracting'):
        # parse the subject index
        subject = re.search('S\d{2}', os.path.basename(filepath)).group(0)
        subject = int(subject.replace('S', ''))

        raw = read_raw_eeglab(filepath, verbose='CRITICAL')
        data, _ = raw[:, :]
        # used for padding later
        max_series_len = max(max_series_len, raw.n_times - cut_from)

        channels_data, channels_locs = [], []
        for channel_idx, channel_data in enumerate(data):
            channel_data = channel_data[cut_from:]
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
        task_types.append(get_task_type(filepath))

    sfreq = raw.info['sfreq']
    return subjects, subjects_data, task_types, locs, max_series_len, max_val, sfreq


def process_data(data_and_locs, max_series_len=None, max_val=None, length_limit=None):
    """
    Normalizes and pads the data.
    :param data_and_locs: tuple. list of rows of data to be processed, and list of corresponding channel locations
    :param max_series_len: integer.
        Maximal length of a series of data. All rows of data (and for each channel) will be
        padded with 0 to this length.
    :param max_val: float.
        All values in the data will be divided by this.
    :param length_limit: integer.
        If set, all sequences longer than this, will be split into smaller chunks of length equal to this parameter.
        The list of locations will also be updated.
        NOTE: setting this value overrides the value for max_series_len for padding the sequences.
    :return: processed data, and updated (if length_limit was set) locations
    """
    data = data_and_locs[0]
    locs = data_and_locs[1]
    max_val = None
    if max_val is not None:
        for row in tqdm(data, desc='Normalizing'):
            row /= max_val
    if length_limit is not None:
        split_data, new_locs = [], []
        for row_idx, row in tqdm(enumerate(data), desc='Splitting'):
            num_splits = np.math.ceil(row.shape[1] / length_limit)
            for i in range(num_splits):
                split = row[:, i * length_limit: min((i + 1) * length_limit, row.shape[1])]
                split_data.append(split)
                # save the same location for the splits
                new_locs.append(locs[row_idx])
        data = split_data
        locs = new_locs
        # if length_limit is set, set the maximum length, to which the sequences will be padded, to length_limit
        max_series_len = length_limit
    if max_series_len is not None:
        data = [np.pad(row, ((0, 0), (0, max_series_len - row.shape[1])), 'constant') for row in
                tqdm(data, desc='Padding')]

    return data, locs


def normalize_labels(labels):
    """
    Transforms labels of arbitrary values into integer classes.
    :param labels: array of labels
    :return array of values from 0 to N-1, where N is the total number of different labels
    """
    labels_set = set(labels)
    labels_mapping = {}
    for label_idx, label in enumerate(labels_set):
        labels_mapping[label] = label_idx
    return [labels_mapping[label] for label in labels]


def save_to_h5(h5_filepath, labels, task_types, locs, data, normalize_images):
    """
    Saves the data to a .hdf5 file with the necessary attributes - total number of unique labels,
    and dimensionality of the data (num_channels x max_series_len).
    :param h5_filepath: path to the output file
    :param labels: list of integers indicating labels for each row of data
    :param locs: list of lists of 3-dimensional locations of channels
    :param data: list of rows of data, where each row should have dimensionality: num_channels x max_series_len
    :param normalize_images: boolean, indicates whether the images have been normalized
    """
    def code_task_types(task_types):
        coded = []
        for task_type in task_types:
            if task_type == 'T_MR':
                coded.append(0)
            elif task_type == 'SN_MR':
                coded.append(1)
            elif task_type == 'A_MR':
                coded.append(2)
            elif task_type == 'T_DR':
                coded.append(3)
            elif task_type == 'SN_DR':
                coded.append(4)
            elif task_type == 'A_DR':
                coded.append(5)
            else:
                raise ValueError('Unknown task type: {}'.format(task_type))
        return coded

    hdf5_file = h5py.File(h5_filepath, 'w')
    hdf5_data = hdf5_file.create_dataset('data', data=data)
    hdf5_labels = hdf5_file.create_dataset('labels', data=labels)
    hdf5_file.create_dataset('locs', data=locs)
    task_types = code_task_types(task_types)
    task_types = np.array(task_types)
    hdf5_file.create_dataset('task_types', data=task_types)

    hdf5_data.attrs['dims'] = data[0].shape
    hdf5_data.attrs['normalized'] = normalize_images
    hdf5_labels.attrs['num_labels'] = len(set(labels))
    hdf5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data extraction')
    parser.add_argument('-d', '--data_folder', default='data.import')
    parser.add_argument('-t', '--target', default='data/extracted.hdf5')
    parser.add_argument('-i', '--images', action='store_true')
    parser.add_argument('-n', '--normalize_images', action='store_true')
    parser.add_argument('-f', '--fft_window_len', default=0.5,
                        help='Size of the window for FFT (in seconds) - used when creating images')
    parser.add_argument('-s', '--single_frame', action='store_true',
                        help="""Whether to perform FFT on the whole series, producing only one image
                        (AKA the 'Single Frame Approach')""")
    parser.add_argument('-l', '--length_limit', default=None, type=int, help="""If set, limits the extracted sequences to the given value.
        Longer sequences are then split into smaller ones.""")
    parser.add_argument('-c', '--cut_from', default=0, type=int, help="""Specifies from which index
     to start reading the data.""")
    args = parser.parse_args()

    target_dir = os.path.dirname(args.target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = find_files(args.data_folder)
    files_found = len(files)
    if files_found < 1:
        print('No files found. Exiting')
        exit()

    print('Files found: {}'.format(files_found))

    subjects, subjects_data, task_types, locs, max_series_len, max_val, sfreq = extract_raw(filepaths=files)
    subjects_data, locs = process_data(
        data_and_locs=(subjects_data, locs),
        max_series_len=max_series_len,
        max_val=max_val,
        length_limit=args.length_limit,
    )
    print(len(set(task_types)))
    print(len(set(subjects)))
    if args.images or True:
        images_data = []
        for row_idx, subject_data in enumerate(tqdm(subjects_data, desc='Images')):
            row_images = raw_to_image(
                raw_data=subject_data,
                locs_3d=locs[row_idx],
                sfreq=sfreq,
                normalize=args.normalize_images or True,
                window_len=args.fft_window_len,
                single_frame=args.single_frame,
            )
            images_data.append(row_images)
        subjects_data = images_data

    print('Saving to h5')
    labels = normalize_labels(subjects)
    save_to_h5(
        h5_filepath=args.target,
        labels=labels,
        task_types=task_types,
        locs=locs,
        data=subjects_data,
        normalize_images=args.normalize_images,
    )
    print('Finished.')
