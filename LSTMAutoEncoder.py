__author__ = 'Steffen'

from ConvolutionalAutoEncoder import ConvolutionalAutoEncoder
from DenoisingConvolutionalAutoEncoder import DenoisingConvolutionalAutoEncoder
from utils import eeg_mse
from keras import backend as K

from keras.engine import Model
from keras.layers import Flatten, TimeDistributed, LSTM, Bidirectional, Input, RepeatVector, Reshape, Dense, \
    Activation, Permute, Lambda, merge


class LSTMAutoEncoder:

    def init_model(self, seq_len, denoising=True, load_conv_weights=True, path=None, train_conv=True, compile=True,
                   attention=True, bidirectional=True):

        image_shape = (3, 32, 32)
        HIDDEN_SIZE = 128

        if denoising:
            cae = DenoisingConvolutionalAutoEncoder()
        else:
            cae = ConvolutionalAutoEncoder()
        if load_conv_weights:
            if path:
                cae.load_from_weights(path)
            else:
                cae.load_from_weights()

        encoder = cae.get_encoder()

        inputs = Input(shape=tuple((seq_len,) + image_shape))
        x = TimeDistributed(Permute((2,3,1)))(inputs)

        # encoder
        for l in encoder.layers:
            # do we want to train this layer
            l.trainable = train_conv
            x = TimeDistributed(l)(x)
        x = TimeDistributed(Flatten())(x)

        if bidirectional:
            activations = Bidirectional(LSTM(HIDDEN_SIZE, activation='tanh', return_sequences=attention), merge_mode='concat')(x)
        else:
            activations = LSTM(HIDDEN_SIZE, activation='tanh', return_sequences=attention)(x)

        if attention:
            # attention mechanism
            x = Flatten()(activations)
            x = Dense(seq_len, activation='tanh')(x)
            x = Activation('softmax')(x)
            x = RepeatVector(HIDDEN_SIZE * 2)(x)
            x = Permute([2, 1])(x)
            attn = merge([activations, x], mode='mul')
            attn = Lambda(lambda x: K.sum(x, axis=-2), output_shape=(HIDDEN_SIZE * 2,))(attn)

            self.Encoder = Model(inputs, attn)
        else:
            self.Encoder = Model(inputs, activations)

        # decoder
        dec_input = Input(shape=(HIDDEN_SIZE * (2 if bidirectional else 1),))
        y = RepeatVector(seq_len)(dec_input)

        y = Bidirectional(LSTM(HIDDEN_SIZE, activation='tanh', return_sequences=True), merge_mode='concat')(y)

        decoder = cae.get_decoder()

        y = TimeDistributed(Dense(2048))(y)
        y = TimeDistributed(Reshape((128,4,4)))(y)

        for l in decoder.layers:
            # we don't want to train this layer
            l.trainable = train_conv
            y = TimeDistributed(l)(y)

        self.Decoder = Model(dec_input, y)

        model_inputs = Input(shape=tuple((seq_len,) + image_shape))
        m = self.Encoder(model_inputs)
        m = self.Decoder(m)

        lstm_ae = Model(model_inputs, m)
        if compile:
            lstm_ae.compile(optimizer='adadelta', loss=eeg_mse)

        self.Model = lstm_ae

    def get_encoder(self):
        return self.Encoder

    def get_final_model(self, seq_len, num_classes, use_weights=True):
        if use_weights:
            self.Model.load_weights('lstmae_weights.h5')
        image_shape = (3, 32, 32)
        input = Input(shape=tuple((seq_len,) + image_shape))
        m = self.Encoder(input)
        m = Dense(num_classes, activation='softmax')(m)

        model = Model(input, m)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def load_from_weights(self, weights_path='lstmae_weights.h5'):
        self.init_model(334, compile=False)
        self.Model.load_weights(weights_path)

import h5py
import numpy as np

if __name__ == '__main__':
    h5_file = '/data/pgram_norm.hdf5'
    data = []
    with h5py.File(h5_file) as f:
        data = np.array(f['data'])
    lstm = LSTMAutoEncoder()
    lstm.init_model(334, denoising=False, compile=True, train_conv=True)
    lstm.Model.fit(data, data, validation_split=0.1, epochs=100, batch_size=16)
    lstm.Model.save_weights('lstmae_weights.h5')
