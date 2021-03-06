__author__ = 'Steffen'

import numpy as np
from keras.layers import Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras.models import load_model
import random
import h5py
from utils import eeg_mse


class ConvolutionalAutoEncoder(object):

    def __init__(self, filter_nums=[32, 64, 128]):
        self.Model, self.Encoder, self.Decoder = ConvolutionalAutoEncoder.model(filter_nums=filter_nums)
        self.maximum = 0
        self.minimum = 0
        self.training_data = []

    @staticmethod
    def model(filter_nums=[32, 64, 128], compile_model=True):
        # 32x32 x 3 channels

        encoder = Sequential()
        image_shape = (32, 32, 3)
        # use as first layer to simplify Sequential model, doesn't actually do anything
        encoder.add(Reshape(image_shape, input_shape=image_shape))

        # values for parameters (filter size, stride etc.) taken from VGG image recognition network
        # encoder
        for idx, filters in enumerate(filter_nums):
            encoder.add(Conv2D(filters, (3, 3), activation='elu', padding='same', name='enc_conv_' + str(idx)))
            encoder.add(MaxPooling2D((2,2), strides=(2, 2), padding='same', name='enc_max_' + str(idx)))


        out_shape = (4, 4) + (filter_nums[-1],)
        decoder = Sequential()
        decoder.add(Reshape(out_shape, input_shape=out_shape))

        # decoder: encoder in reverse
        for idx, filters in enumerate(reversed(filter_nums)):
            decoder.add(Conv2D(filters, (3, 3), activation='elu', padding='same', name='dec_conv_' + str(idx)))
            decoder.add(UpSampling2D((2,2), name='dec_up_' + str(idx)))

        # last convolution for output layer
        decoder.add(Conv2D(3, (3, 3), activation='linear', padding='same', name='dec_conv_out' + str(idx)))

        autoencoder = Sequential()
        autoencoder.add(Reshape(image_shape, input_shape=image_shape))
        autoencoder.add(encoder)
        autoencoder.add(decoder)

        # Don't compile if loading weights for predictions
        if compile_model:
            autoencoder.compile(optimizer='adadelta', loss=eeg_mse)
        return autoencoder, encoder, decoder

    def scale(self, images, min=0, max=1.0):
        """
        Scale images to range [0, 1]
        :param images: numpy ndarray of values to scale
        """
        self.maximum = max
        self.minimum = min
        return (images - self.minimum) / (self.maximum - self.minimum)

    def descale(self, images):
        """
        Scale images back to original range after scaling
        :param images: numpy ndarray of values to descale
        :return:
        """
        return (self.maximum - self.minimum) * images + self.minimum

    def train(self, data_train, epochs=10):
        self.Model.fit(data_train, data_train, validation_split=0.1, epochs=epochs, batch_size=32)

    @staticmethod
    def extract_data(h5_file):
        with h5py.File(h5_file) as f:
            data = np.array(f['data'])
            total_images = data.shape[0] * data.shape[1]
            # swap dimensions for the network: last dim must be channels
            # also, merge all subjects into one dim
            data = np.reshape(np.rollaxis(data, 2, 5), (total_images, 32, 32, 3))
            data = np.array([x for x in data if x.any()])
            #data = self.scale(data, min=data.min(), max=data.max())
        return data

    def train_from_dataset(self, h5_file='/data/pgram_norm.hdf5', epochs=100, use_data=1.0):
        self.training_data = ConvolutionalAutoEncoder.extract_data(h5_file)
        if use_data < 1.0:
            self.training_data = np.array([self.training_data[i]
                                           for i in sorted(random.sample(xrange(len(self.training_data)),
                                                                int(use_data * len(self.training_data))))])
        self.train(self.training_data, epochs)

    def save(self, path='cae_saved.h5'):
        self.Model.save(path)

    def load(self, path='cae_saved.h5'):
        self.Model = load_model(path)

    def load_from_weights(self, path='cae_weights.h5'):
        self.Model, self.Encoder, self.Decoder = ConvolutionalAutoEncoder.model(compile_model=False)
        self.Model.load_weights(path)

    def save_weights(self, path='cae_weights.h5'):
        self.Model.save_weights(path)

    def get_decoder(self):
        return self.Decoder

    def get_encoder(self):
        return self.Encoder
