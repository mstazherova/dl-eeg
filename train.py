import argparse
import os

import datetime
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import models
from eeglearn_bench.data import *

CHECKPOINTS_DIR = './checkpoints'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def make_dirs(*args):
    import os
    if not os.path.isdir(CHECKPOINTS_DIR):
        print('Creating dir: {}'.format(CHECKPOINTS_DIR))
        os.makedirs(CHECKPOINTS_DIR)
    for path in args:
        if not os.path.isdir(path):
            print('Creating dir: {}'.format(path))
            os.makedirs(path)


def train(args):
    # load the data
    print('Loading data')
    if args.kfold:
        data_wrapper = KFoldDataWrapper(
            dataset_path=args.train_data,
            k=args.folds,
        )
    elif args.subject_out:
        data_wrapper = SubjectOutDataWrapper(
            dataset_path=args.train_data,
            k=args.folds,
        )
        args.kfold = True
    else:
        data_wrapper = DataWrapper(
            dataset_path=args.train_data,
            val_data_perc=args.val_data,
            test_data_perc=args.test_data
        )


    num_runs = args.folds if args.kfold else 1
    accuracies = [0 for _ in range(num_runs)]
    for run_idx in range(num_runs):
        print('\n####### RUN {} ##########\n'.format(run_idx))

        # Create the model
        model = getattr(models, args.model_name)
        model = model(
            num_classes=data_wrapper.num_classes,
            input_shape=data_wrapper.data_dim,
        )
        if run_idx < 1:
            print(model.summary())

        checkpoints_filepath = os.path.join(CHECKPOINTS_DIR,
                                            '{}_{}{}.hdf5'.format(model.name,
                                                                  datetime.datetime.now().strftime("%Y%m%d_%H%M"),
                                                                  '_{}'.format(run_idx) if args.kfold else '',
                                                                  ))
        tensorboard_path = './tensorboard/{}_{}{}'.format(
            model.name,
            datetime.datetime.now().strftime("%Y%m%d_%H%M"),
            '_{}'.format(run_idx) if args.kfold else '',
        )
        if not args.no_save:
            make_dirs(tensorboard_path)
        callbacks = [] if args.no_save else [
            TensorBoard(log_dir='./tensorboard', write_images=True),
            ModelCheckpoint(filepath=checkpoints_filepath,
                            verbose=1,
                            save_best_only=True,
                            monitor='val_acc'),
        ]

        if args.kfold:
            data_wrapper.load_fold(run_idx)

        # Train the model
        model.fit_generator(
            data_wrapper.gen_data(batch_size=args.batch_size),
            epochs=args.epochs,
            steps_per_epoch=args.steps_train,
            validation_data=data_wrapper.gen_data(val=True, batch_size=args.batch_size_val),
            validation_steps=args.steps_val,
            callbacks=callbacks,
        )

        # Evaluate the model
        acc, total = 0, 0
        if args.kfold:
            gen = data_wrapper.gen_data(loop=False, val=True, shuffle=False, batch_size=1)
        else:
            gen = data_wrapper.gen_data(loop=False, test=True, shuffle=False, batch_size=1)
        # load the best model
        try:
            model.load_weights(checkpoints_filepath)
            for x, y in gen:
                y_ = model.predict(x)
                acc += 1 if y_.argmax() == y.argmax() else 0
                total += 1
            accuracy = acc * 100 / total
            accuracies[run_idx] = accuracy
            print("Accuracy: {}%".format(accuracy))
        except OSError:
            print('No model file found. The model probably didnt learn anything')

    if args.kfold:
        # print accuracies for each fold
        print(accuracies)
        print('Average: {}'.format(sum(accuracies) / args.folds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEGLearn benchmarks')
    parser.add_argument('-t', '--train_data', default='../data/pgram_norm.hdf5')
    parser.add_argument('-m', '--model_name', default='bi_lstm')
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_val', type=int, default=32)
    parser.add_argument('--steps_train', type=int, default=21)
    parser.add_argument('--steps_val', type=int, default=3)
    parser.add_argument('--val_data', type=float, default=0.125)
    parser.add_argument('--test_data', type=float, default=0.125)
    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--subject_out', action='store_true')
    parser.add_argument('--folds', type=int, default=8)
    parser.add_argument('--no_save', action='store_true',
                        help='If set, no tensorboard information or model weights will be saved')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    args = parser.parse_args()

    train(args)
