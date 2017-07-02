__author__ = 'Steffen'

from ConvolutionalAutoEncoder import ConvolutionalAutoEncoder
import numpy as np


class DenoisingConvolutionalAutoEncoder(ConvolutionalAutoEncoder):

    def gaussian_noise(self, input_images, mean=0, var=0.0001):
        noisy_images = np.ndarray(input_images.shape)
        # multiply each image by mask to create black border
        for index, image in enumerate(input_images):
            mask = np.sum(image, axis=2) != 0
            mask_channels = np.rollaxis(np.tile(mask, (3, 1, 1)), 0, 3)
            gauss = np.random.normal(mean, np.sqrt(var), image.shape).reshape(image.shape)
            noisy_images[index] = mask_channels * (image + gauss)
        return input_images

    def train(self, data_train, epochs=100):
        noisy_input = self.gaussian_noise(data_train)
        self.Model.fit(noisy_input, data_train, validation_split=0.1, nb_epoch=epochs, batch_size=32)