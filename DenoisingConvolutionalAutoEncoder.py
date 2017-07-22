__author__ = 'Steffen'

from ConvolutionalAutoEncoder import ConvolutionalAutoEncoder
import numpy as np


class DenoisingConvolutionalAutoEncoder(ConvolutionalAutoEncoder):

    def __init__(self, noise_var=0.01, filter_nums=[32, 64, 128]):
        super(DenoisingConvolutionalAutoEncoder, self).__init__(filter_nums)
        self.noise_var = noise_var

    def gaussian_noise(self, input_images, mean=0, var=0.01):
        return input_images + np.random.normal(mean, np.sqrt(var), input_images.shape)

    def train(self, data_train, epochs=100):
        noisy_input = self.gaussian_noise(data_train, var=self.noise_var)
        self.Model.fit(noisy_input, data_train, validation_split=0.1, epochs=epochs, batch_size=32)

    def save(self, path='dcae_saved.h5'):
        ConvolutionalAutoEncoder.save(self, path)

    def load(self, path='dcae_saved.h5'):
        ConvolutionalAutoEncoder.load(self, path)

    def save_weights(self, path='dcae_weights.h5'):
        ConvolutionalAutoEncoder.save_weights(self, path)

    def load_from_weights(self, path='dcae_weights.h5'):
        ConvolutionalAutoEncoder.load_from_weights(self, path)