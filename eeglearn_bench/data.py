import random

import h5py
import numpy as np


class DataWrapper:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        h5_file = h5py.File(self.dataset_path, 'r')
        self.num_classes = h5_file['labels'].attrs['num_labels']
        self.data_dim = h5_file['data'].attrs['dims']

    def gen_data(self, shuffle=True, loop=True):
        h5_file = h5py.File(self.dataset_path, 'r')
        while True:
            data, labels = h5_file['data'], h5_file['labels']
            data_and_labels = [(x, y) for x, y in zip(data, labels)]

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
    data_wrapper = DataWrapper()
    for x, y in data_wrapper.gen_data(loop=False):
        print(x.shape)
        print(y)
