import random

import h5py
import numpy as np


class DataWrapper:
    def __init__(self, dataset_path, val_data_perc=0.1, test_data_perc=0.0):
        if val_data_perc + test_data_perc >= 1:
            raise ValueError('Sum of test and validation data percentage must be less than 1.')

        self.dataset_path = dataset_path
        h5_file = h5py.File(self.dataset_path, 'r')
        self.num_classes = h5_file['labels'].attrs['num_labels']
        self.data_dim = h5_file['data'].attrs['dims']

        data, labels = h5_file['data'], h5_file['labels']
        data_and_labels = [xy for xy in zip(data, labels)]
        random.shuffle(data_and_labels)

        val_cnt = int(val_data_perc * len(data_and_labels))
        self.val_data = data_and_labels[:val_cnt]
        if test_data_perc == 0:
            # if no amount of test data is specified take the rest of the data as the training data
            self.train_data = data_and_labels[val_cnt:]
        else:
            # index of the last test data sample = idx of the last val data sample + no test samples
            test_idx = int(test_data_perc * len(data_and_labels)) + val_cnt
            self.test_data = data_and_labels[val_cnt:test_idx]
            self.train_data = data_and_labels[test_idx:]

    def gen_data(self, shuffle=True, loop=True, val=False, test=False):
        if val:
            data_and_labels = self.val_data
        elif test:
            data_and_labels = self.test_data
        else:
            data_and_labels = self.train_data

        while True:
            if shuffle:
                random.shuffle(data_and_labels)

            for X, label in data_and_labels:
                X = X.reshape((1,) + tuple(self.data_dim))

                Y = np.zeros(self.num_classes)
                Y[label] = 1
                Y = Y.reshape(1, self.num_classes)

                yield X, Y
            if not loop:
                break


if __name__ == '__main__':
    dataset_path = '/datasets/CogReplay/dl-eeg/pgram_norm.hdf5'
    #dataset_path = '../data/extracted.hdf5'
    data_wrapper = DataWrapper(dataset_path=dataset_path, test_data_perc=0.1)
    for x, y in data_wrapper.gen_data(shuffle=False, loop=False):
        print(x.mean())
        print(y)
