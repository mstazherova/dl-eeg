import random
from collections import defaultdict

import h5py
import numpy as np


class DataWrapper:
    def __init__(self, dataset_path, val_data_perc=0.1, test_data_perc=0.1):
        if val_data_perc + test_data_perc >= 1:
            raise ValueError('Sum of test and validation data percentage must be less than 1.')

        data_and_labels = self._load_data(dataset_path=dataset_path)
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

    def _load_data(self, dataset_path):
        self.dataset_path = dataset_path
        h5_file = h5py.File(self.dataset_path, 'r')
        self.num_classes = h5_file['labels'].attrs['num_labels']
        self.data_dim = h5_file['data'].attrs['dims']

        data, labels = h5_file['data'], h5_file['labels']
        data_and_labels = [xy for xy in zip(data, labels)]

        return data_and_labels

    def gen_data(self, shuffle=True, loop=True, val=False, test=False, batch_size=8):
        if val:
            data_and_labels = self.val_data
        elif test:
            data_and_labels = self.test_data
        else:
            data_and_labels = self.train_data

        items_in_batch = 0
        Xs, Ys = [], []
        while True:
            if shuffle:
                random.shuffle(data_and_labels)

            for X, label in data_and_labels:
                Xs.append(X)

                Y = np.zeros(self.num_classes)
                Y[label] = 1
                Ys.append(Y)

                items_in_batch += 1
                if items_in_batch == batch_size:
                    yield np.array(Xs), np.array(Ys)
                    items_in_batch = 0
                    Xs, Ys = [], []

            if not loop:
                break


class KFoldDataWrapper(DataWrapper):
    def __init__(self, dataset_path, k=10):
        self.k = k
        data_and_labels = self._load_data(dataset_path=dataset_path)
        random.shuffle(data_and_labels)
        self.data_and_labels = data_and_labels

    def load_fold(self, fold_idx):
        if fold_idx < 0 or fold_idx >= self.k:
            raise ValueError("'k' must be in range [0, {})".format(self.k))

        val_set_size = int(len(self.data_and_labels) / self.k)
        self.val_data = self.data_and_labels[val_set_size * fold_idx:val_set_size * (fold_idx + 1)]
        self.train_data = []
        if fold_idx > 0:
            self.train_data = self.data_and_labels[:val_set_size * fold_idx]
        if val_set_size * (fold_idx + 1) < len(self.data_and_labels):
            self.train_data.extend(self.data_and_labels[val_set_size * (fold_idx + 1):])

    def gen_data(self, shuffle=True, loop=True, val=False, test=False, batch_size=8):
        if test:
            raise ValueError('{} does not have a test dataset'.format(self.__class__))
        return super().gen_data(shuffle=shuffle, loop=loop, val=val, test=False, batch_size=batch_size)

    def get_test_set(self):
        raise NotImplemented()


class SubjectOutDataWrapper(KFoldDataWrapper):
    def __init__(self, dataset_path, k=8, take_DR=False):
        self.k = k
        self.take_DR = take_DR
        self.dataset_path = dataset_path
        h5_file = h5py.File(self.dataset_path, 'r')
        self.num_subjects = h5_file['labels'].attrs['num_labels']
        self.num_classes = 6 if take_DR else 3
        self.data_dim = h5_file['data'].attrs['dims']

        self.subjects_data = defaultdict(list)
        data, subjects, tasks = h5_file['data'], h5_file['labels'], h5_file['task_types']
        for x, subj, y in zip(data, subjects, tasks):
            if not take_DR and y > 2:
                continue
            self.subjects_data[subj].append((x, y))

    def load_fold(self, fold_idx):
        if fold_idx < 0 or fold_idx >= self.k:
            raise ValueError("'k' must be in range [0, {})".format(self.k))

        fold_size = int(self.num_subjects / self.k)
        self.val_data = []
        train_idxes = [idx for idx in range(self.num_subjects)]
        for i in range(fold_size):
            self.val_data.extend(self.subjects_data[fold_idx + i])
            train_idxes.remove(i)
        self.train_data = []
        for i in train_idxes:
            self.train_data.extend(self.subjects_data[i])


if __name__ == '__main__':
    dataset_path = '/datasets/CogReplay/dl-eeg/pgram_norm.hdf5'
    dataset_path = '../data/extracted.hdf5'

    data_wrapper = SubjectOutDataWrapper(dataset_path=dataset_path)
    data_wrapper.load_fold(0)
    gen = data_wrapper.gen_data()
    print(next(gen))
    print(next(gen))
    gen = data_wrapper.gen_data(val=True)
    print(next(gen))
    print(next(gen))
    data_wrapper.load_fold(9)
    gen = data_wrapper.gen_data()
    print(next(gen))
    print(next(gen))
    gen = data_wrapper.gen_data(val=True)
    print(next(gen))
    print(next(gen))
    data_wrapper.load_fold(5)
    gen = data_wrapper.gen_data()
    print(next(gen))
    print(next(gen))
    gen = data_wrapper.gen_data(val=True)
    print(next(gen))
    print(next(gen))
    # these should raise an exception
    # data_wrapper.load_fold(10)
    # data_wrapper.load_fold(-1)

    data_wrapper = DataWrapper(dataset_path=dataset_path, test_data_perc=0.1)
    for x, y in data_wrapper.gen_data(shuffle=False, loop=False):
        print(x.mean())
        print(y)
