from keras.layers import Dense, MaxPooling2D, Flatten, TimeDistributed, Conv2D, Dropout, LSTM, Reshape
from keras.models import Sequential

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
