import keras
from keras import Input
from keras.engine import Model
from keras.layers import Dense, MaxPooling2D, Flatten, TimeDistributed, Conv2D, Dropout, LSTM, Reshape, Bidirectional, \
    Conv1D, LeakyReLU
from keras.models import Sequential

from os import sys, path
import os
import inspect
from keras.layers import Input

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from ConvolutionalAutoEncoder import ConvolutionalAutoEncoder

CONV_ACT = 'relu'


def conv(num_classes, input_shape):
    """
    Convolutional clasiffier for single images.
    """
    model = Sequential()
    model.add(Reshape(input_shape[1:], input_shape=input_shape))
    model.add(Conv2D(32, 3, strides=1, padding='same', activation=CONV_ACT))
    model.add(Conv2D(32, 3, strides=1, padding='same', activation=CONV_ACT))
    model.add(Conv2D(32, 3, strides=1, padding='same', activation=CONV_ACT))
    model.add(Conv2D(32, 3, strides=1, padding='same', activation=CONV_ACT))
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th",
        ))
    model.add(Conv2D(64, 3, strides=1, padding='same', activation=CONV_ACT))
    model.add(Conv2D(64, 3, strides=1, padding='same', activation=CONV_ACT))
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th",
        ))
    model.add(Conv2D(128, 3, strides=1, padding='same', activation=CONV_ACT))
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th"
        ))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.add(Dropout(0.5))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.__setattr__('name', 'conv')
    return model


def get_conv(input_shape):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, 3, padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(32, 3, padding='same', activation=CONV_ACT)))
    model.add(TimeDistributed(Conv2D(32, 3, padding='same', activation=CONV_ACT)))
    model.add(TimeDistributed(Conv2D(32, 3, padding='same', activation=CONV_ACT)))
    model.add(TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th",
        )))
    model.add(TimeDistributed(Conv2D(64, 3, padding='same', activation=CONV_ACT)))
    model.add(TimeDistributed(Conv2D(64, 3, padding='same', activation=CONV_ACT)))
    model.add(TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th",
        )))
    model.add(TimeDistributed(Conv2D(128, 3, padding='same', activation=CONV_ACT)))
    model.add(TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th"
        )))
    return model


def maxpool(num_classes, input_shape):
    model = get_conv(input_shape)
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.add(Dropout(0.5))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.__setattr__('name', 'maxpool')
    return model


def lstm(num_classes, input_shape):
    model = get_conv(input_shape)

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, activation='tanh'))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.add(Dropout(0.5))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.__setattr__('name', 'lstm')
    return model


def bi_lstm(num_classes, input_shape):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, 3, 3, activation='relu', border_mode='same'), input_shape=input_shape))
    model.add(TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')
    ))
    model.add(TimeDistributed(Conv2D(64, 3, 3, activation='relu', border_mode='same')))
    model.add(TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')
    ))
    model.add(TimeDistributed(Conv2D(128, 3, 3, activation='relu', border_mode='same')))
    model.add(TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')
    ))
    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, activation='tanh')))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.__setattr__('name', 'bi_lstm')
    return model


def bi_lstm_weights(num_classes, input_shape, weights_path='../cae_weights.h5'):
    cae = ConvolutionalAutoEncoder()
    cae.load_from_weights(path=weights_path)
    encoder = cae.get_encoder()

    inputs = Input(shape=tuple(input_shape))
    x = inputs
    for l in encoder.layers:
        x = TimeDistributed(l)(x)
    x = TimeDistributed(Flatten())(x)

    x = Bidirectional(LSTM(128, activation='tanh'))(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.__setattr__('name', 'bi_lstm')
    return model


def mixed_paper(num_classes, input_shape):
    def CONV_ACT():
        l = LeakyReLU()
        l.__name__ = 'leakyrelu'
        return l

    inputs = Input(shape=tuple(input_shape))
    x = TimeDistributed(Conv2D(32, 3, border_mode='same', activation=CONV_ACT()))(inputs)
    x = TimeDistributed(Conv2D(32, 3, border_mode='same', activation=CONV_ACT()))(x)
    x = TimeDistributed(Conv2D(32, 3, border_mode='same', activation=CONV_ACT()))(x)
    x = TimeDistributed(Conv2D(32, 3, border_mode='same', activation=CONV_ACT()))(x)
    x = TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th",
        ))(x)
    x = TimeDistributed(Conv2D(64, 3, border_mode='same', activation=CONV_ACT()))(x)
    x = TimeDistributed(Conv2D(64, 3, border_mode='same', activation=CONV_ACT()))(x)
    x = TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th",
        ))(x)
    x = TimeDistributed(Conv2D(128, 3, border_mode='same', activation=CONV_ACT()))(x)
    convs = TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th"
        ))(x)

    lstm = TimeDistributed(Flatten())(convs)
    lstm = LSTM(
        128
        , activation='tanh',
        return_sequences=False,
    )(lstm)

    conv_1d = Reshape((int(convs.shape[-1]), int(convs.shape[-4]) * int(convs.shape[-2] * convs.shape[-3])))(convs)
    conv_1d = Conv1D(64, 3)(conv_1d)
    conv_1d = Flatten()(conv_1d)

    x = keras.layers.concatenate([lstm, conv_1d])
    x = Dense(256, activation=CONV_ACT())(x)
    x = Dense(num_classes, activation='softmax', )(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.__setattr__('name', 'mixed')
    return model


def mixed(num_classes, input_shape):
    def CONV_ACT():
        l = LeakyReLU()
        l.__name__ = 'leakyrelu'
        return l

    inputs = Input(shape=tuple(input_shape))
    x = TimeDistributed(Conv2D(32, 3, border_mode='same', activation=CONV_ACT()))(inputs)
    x = TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th",
        ))(x)
    x = TimeDistributed(Conv2D(64, 3, border_mode='same', activation=CONV_ACT()))(x)
    x = TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th",
        ))(x)
    x = TimeDistributed(Conv2D(128, 3, border_mode='same', activation=CONV_ACT()))(x)
    convs = TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            dim_ordering="th"
        ))(x)

    lstm = TimeDistributed(Flatten())(convs)
    lstm = LSTM(128, activation='tanh')(lstm)

    conv_1d = Reshape((int(convs.shape[-1]), int(convs.shape[-4]) * int(convs.shape[-2] * convs.shape[-3])))(convs)
    conv_1d = Conv1D(64, 3)(conv_1d)
    conv_1d = Flatten()(conv_1d)

    x = keras.layers.concatenate([lstm, conv_1d])
    x = Dense(256, activation=CONV_ACT())(x)
    x = Dense(num_classes, activation='softmax', )(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.__setattr__('name', 'mixed')
    return model

