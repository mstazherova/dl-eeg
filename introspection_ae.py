"""
Script for computing introspection - input yielding the maximum activation of a filter of a trained model.
"""

from keras import backend as K
from keras.models import load_model
import numpy as np
import argparse

#from lib.encoder.autoencoder import SAMPLE_RATE
#from lib.encoder.utils import plot_mel_spectrogram

MODEL_FILE_NAME = 'encoder.h5'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Introspection')
    parser.add_argument('-m', '--model_file_name', default=MODEL_FILE_NAME)
    parser.add_argument('-f', '--filter_indexes', type=int, nargs='+', default=[50])
    parser.add_argument('-l', '--layer_index', type=int, default=-1)
    args = parser.parse_args()

    model = load_model(args.model_file_name)

    model_input = model.layers[0].input
    input_shape = model_input.get_shape()
    input_width, input_height = input_shape[1], input_shape[2]

    filter_indexes = args.filter_indexes

    for filter_index in filter_indexes:
        print('Computing introspection for filter {} of model {}'.format(filter_index, args.model_file_name))
        # build a loss function that maximizes the activation of the last layer
        layer_output = model.layers[args.layer_index].output
        loss = K.mean(layer_output[:, :, :, filter_index])
        # compute the gradient of the input wrt this loss
        grads = K.gradients(loss, model_input)[0]
        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        # this function returns the loss and grads given the input data
        iterate = K.function([model_input], [loss, grads])

        input_mel_data = np.random.random((1, input_width, input_height, 1))

        for i in range(20):
            loss_value, grads_value = iterate([input_mel_data])
            input_mel_data += grads_value

        log_S = input_mel_data.reshape(input_width, input_height)
        plot_mel_spectrogram(sr=SAMPLE_RATE, log_S=log_S,
                             file_to_save='{}_layer_{}_filter_{}_introspection'.format(
                                 args.model_file_name.split('.')[0],
                                 args.layer_index,
                                 filter_index))