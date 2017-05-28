__author__ = 'Steffen'

import numpy as np
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import h5py

class ConvolutionalAutoEncoder:

    def __init__(self):
        self.Model = ConvolutionalAutoEncoder.model()

    @staticmethod
    def model():
        # 32x32 x 3 channels
        image_shape = (32, 32, 3)
        input_img = Input(shape=image_shape)

        # values for parameters (filter size, stride etc.) taken from VGG image recognition network
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2,2), strides=(2, 2))(x)
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)

        encoded = MaxPooling2D((2,2), strides=(2, 2))(x)

        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(encoded)
        x = UpSampling2D((2,2))(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2,2))(x)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2,2))(x)
        decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder

    def train(self, data_train, epochs=100):
        self.Model.fit(data_train, data_train, validation_split=0.1, nb_epoch=epochs, batch_size=32)

    def train_from_dataset(self, h5_file='/datasets/CogReplay/dl-eeg/extracted_images.hdf5', epochs=100):
        with h5py.File(h5_file) as f:
            data = np.array(f['data'])
            # swap dimensions for the network: last dim must be channels
            # also, merge all subjects into one dim
            data = np.reshape(np.rollaxis(data, 2, 5), (7660, 32, 32, 3))
        self.train(data, epochs)