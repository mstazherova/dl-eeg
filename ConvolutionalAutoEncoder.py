__author__ = 'Steffen'

import numpy as np
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras.models import load_model
import h5py


class ConvolutionalAutoEncoder:

    def __init__(self):
        self.Model = ConvolutionalAutoEncoder.model()
        self.maximum = 0
        self.minimum = 0
        self.training_data = []

    @staticmethod
    def model(num_layers=3, filter_nums=[64, 128, 256], compile_model=True):
        # 32x32 x 3 channels

        autoencoder = Sequential()
        image_shape = (32, 32, 3)
        autoencoder.add(Input(shape=image_shape))

        # values for parameters (filter size, stride etc.) taken from VGG image recognition network
        # encoder
        for i in range(num_layers):
            autoencoder.add(Convolution2D(filter_nums[i], 3, 3, activation='relu', border_mode='same'))
            autoencoder.add(MaxPooling2D((2,2), strides=(2, 2), border_mode='same'))

        # decoder: encoder in reverse
        for i in range(num_layers).reverse():
            autoencoder.add(Convolution2D(filter_nums[i], 3, 3, activation='relu', border_mode='same'))
            autoencoder.add(UpSampling2D((2,2)))

        # last convolution for output layer
        autoencoder.add(Convolution2D(3, 3, 3, activation='linear', border_mode='same'))

        # Don't compile if loading weights for predictions
        if compile_model:
            autoencoder.compile(optimizer='adadelta', loss='mse')

        return autoencoder

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
        self.Model.fit(data_train, data_train, validation_split=0.1, nb_epoch=epochs, batch_size=32)

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

    def train_from_dataset(self, h5_file='/datasets/CogReplay/dl-eeg/pgram_norm.hdf5', epochs=100):
        self.training_data = ConvolutionalAutoEncoder.extract_data(h5_file)
        self.train(self.training_data, epochs)

    def save(self, path='cae_saved.h5'):
        self.Model.save(path)

    def load(self, path='cae_saved.h5'):
        self.Model = load_model(path)

    def load_from_weights(self, path='cae_weights.h5'):
        self.Model = ConvolutionalAutoEncoder.model(compile_model=False)
        self.Model.load_weights(path)

    def save_weights(self, path='cae_weights.h5'):
        self.Model.save_weights(path)

    def get_encoder(self):
        # remove decoder layers
        encoder = self.Model
        for i in range(0, 7):
            encoder.layers.pop()
        # fix output layer
        encoder.outputs = [encoder.layers[-1].output]
        encoder.layers[-1].outbound_nodes = []
        return encoder