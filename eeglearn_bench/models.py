from keras.layers import Dense, MaxPooling2D, Flatten, TimeDistributed, Conv2D, Dropout, LSTM, Reshape, Bidirectional
from keras.models import Sequential

from ..ConvolutionalAutoEncoder import ConvolutionalAutoEncoder

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

    model.add(TimeDistributed(Conv2D(64, 3, 3, activation='relu', border_mode='same')))
    model.add(TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')
    ))
    model.add(TimeDistributed(Conv2D(128, 3, 3, activation='relu', border_mode='same')))
    model.add(TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')
    ))
    model.add(TimeDistributed(Conv2D(256, 3, 3, activation='relu', border_mode='same')))
    model.add(TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')
    ))
    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, activation='tanh'), input_shape))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.__setattr__('name', 'bi_lstm')
    return model

def bi_lstm_weights(num_classes, input_shape):
    model = Sequential()

    cae = ConvolutionalAutoEncoder()
    cae.load_from_weights()
    encoder = cae.get_encoder()

    model.add(TimeDistributed(encoder))
    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, activation='tanh'), input_shape))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.__setattr__('name', 'bi_lstm')
    return model