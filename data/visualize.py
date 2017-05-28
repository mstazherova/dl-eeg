import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt


def plot_image(image_data, save_path=None):
    assert image_data.shape[0] == 3

    fig = plt.figure()
    plt.axis('off')

    for channel_id in range(image_data.shape[0]):
        channel_data = image_data[channel_id]
        channel_data = channel_data.reshape((image_data.shape[1], image_data.shape[2]))

        fig.add_subplot(1, 4, channel_id + 1)
        plt.axis('off')
        plt.imshow(
            channel_data,
            vmin=0,
            vmax=channel_data.max(),
        )

    # somehow just using np.reshape to transform the data to image was transforming it incorrectly - thus
    # we have to do it manually
    colored_frame_image = np.zeros((image_data.shape[1], image_data.shape[2], image_data.shape[0]))
    for row_id in range(image_data.shape[1]):
        for col_id in range(image_data.shape[2]):
            for channel_id in range(image_data.shape[0]):
                colored_frame_image[row_id, col_id, channel_id] = image_data[channel_id, row_id, col_id]

    fig.add_subplot(1, 4, 4)
    plt.axis('off')
    plt.imshow(
        colored_frame_image,
        vmin=0,
        vmax=colored_frame_image.max()
    )

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data extraction')
    parser.add_argument('-f', '--hdf5_filepath', default='images.hdf5')
    parser.add_argument('-s', '--save_path', default=None,
                        help='If specified, saves the image to the given path instead of displaying')
    parser.add_argument('ROW_IDX', type=int, nargs='?', default=0, help='Index of the row of the dataset')
    parser.add_argument('IMG_IDX', type=int, nargs='?', default=0, help='Index of the image in the sequence of the row')
    args = parser.parse_args()

    h5_file = h5py.File(args.hdf5_filepath, 'r')

    images = np.array(h5_file['data'])
    plot_image(
        image_data=images[args.ROW_IDX][args.IMG_IDX],
        save_path=args.save_path,
    )
