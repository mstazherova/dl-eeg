from keras.layers import Dense, MaxPooling2D, Flatten, TimeDistributed, Conv2D, Dropout
from keras.models import Sequential


def maxpool(num_classes, input_shape):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, 3, padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(32, 3, padding='same')))
    model.add(TimeDistributed(Conv2D(32, 3, padding='same')))
    model.add(TimeDistributed(Conv2D(32, 3, padding='same')))
    model.add(TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            dim_ordering="th",
        )))
    model.add(TimeDistributed(Conv2D(64, 3, padding='same')))
    model.add(TimeDistributed(Conv2D(64, 3, padding='same')))
    model.add(TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            dim_ordering="th",
        )))
    model.add(TimeDistributed(Conv2D(128, 3, padding='same')))
    model.add(TimeDistributed(
        MaxPooling2D(
            pool_size=(2, 2),
            dim_ordering="th"
        )))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.add(Dropout(0.5))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.__setattr__('name', 'maxpool')
    return model
