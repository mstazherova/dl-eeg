__author__ = 'Steffen'

from ConvolutionalAutoEncoder import ConvolutionalAutoEncoder
from DenoisingConvolutionalAutoEncoder import DenoisingConvolutionalAutoEncoder
from utils import eeg_mse
from keras import backend as K

from keras.engine import Model
from keras.layers import Flatten, TimeDistributed, LSTM, Bidirectional, Input, RepeatVector, Reshape, Dense, \
    Activation, Permute, Lambda, merge


class LSTMAutoEncoder:

    def init_model(self, seq_len, denoising=True, load_conv_weights=True, path=None, train_conv=True, compile=True):

        image_shape = (32, 32, 3)
        HIDDEN_SIZE = 256

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
        x = inputs

        # encoder
        for l in encoder.layers:
            # do we want to train this layer
            l.trainable = train_conv
            x = TimeDistributed(l)(x)
        x = TimeDistributed(Flatten())(x)

        activations = Bidirectional(LSTM(HIDDEN_SIZE, activation='tanh', return_sequences=True), merge_mode='concat')(x)

        # attention mechanism
        x = Flatten()(activations)
        x = Dense(seq_len, activation='tanh')(x)
        x = Activation('softmax')(x)
        x = RepeatVector(HIDDEN_SIZE * 2)(x)
        x = Permute([2, 1])(x)
        attn = merge([activations, x], mode='mul')
        attn = Lambda(lambda x: K.sum(x, axis=-2), output_shape=(HIDDEN_SIZE * 2,))(attn)

        self.Encoder = Model(inputs, attn)

        # decoder
        dec_input = Input(shape=(HIDDEN_SIZE * 2,))
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
            self.load_from_weights()
        image_shape = (32, 32, 3)
        input = Input(shape=tuple((seq_len,) + image_shape))
        m = self.Encoder(input)
        m = Dense(num_classes, activation='softmax')(m)

        model = Model(input, m)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def load_from_weights(self, weights_path='lstmae_weights.h5'):
        self.init_model(334, compile=False)
        self.Model.load_weights(weights_path)