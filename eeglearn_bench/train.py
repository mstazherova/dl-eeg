import argparse
import os

import datetime
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from eeglearn_bench import models
from eeglearn_bench.data import DataWrapper

CHECKPOINTS_DIR = './checkpoints'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def make_dirs():
    import os
    if not os.path.isdir(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)


def train(train_data_path):
    test_data_path = ''

    data_wrapper = DataWrapper(dataset_path=train_data_path)
    # Create the model
    model = models.lstm(
        num_classes=data_wrapper.num_classes,
        input_shape=data_wrapper.data_dim,
    )
    print(model.summary())

    # Train the model
    checkpoints_filepath = os.path.join(CHECKPOINTS_DIR,
                                        '{}_{}.hdf5'.format(model.name, datetime.datetime.now().strftime("%Y%m%d_%H%M")
                                                            ))
    callbacks = [TensorBoard(log_dir='./tensorboard', write_images=True),
                 ModelCheckpoint(filepath=checkpoints_filepath,
                                 verbose=1,
                                 save_best_only=True),
                 ]
    model.fit_generator(
        data_wrapper.gen_data(),
        epochs=1500,
        steps_per_epoch=64,
        validation_data=data_wrapper.gen_data(),
        validation_steps=16,
        callbacks=callbacks,
    )
    # Evaluate the model
    scores = model.evaluate_generator(data_wrapper.gen_data(), steps=1000)
    print("Accuracy: {}%".format(scores[1] * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEGLearn benchmarks')
    parser.add_argument('-t', '--train_data', default='../data/extracted.hdf5')
    args = parser.parse_args()

    make_dirs()
    train(
        train_data_path=args.train_data,
    )