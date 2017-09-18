from keras import objectives
from keras import backend as K

import numpy as np

# hack neccessary to make matplotlib run on medusa
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
# end of hack
import matplotlib.pyplot as plt


PLOTS_DIR = 'plots'


def plot_fft(fft_data):
    plt.plot(abs(fft_data[:(len(fft_data) / 2 - 1)]), 'r')
    plt.show()


def plot_spectrogram(t, f, Sxx):
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_mel_spectrogram(sr, log_S, file_to_save=None, block=True):
    # Make a new figure
    plt.figure(figsize=(12, 8))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    import librosa
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    if file_to_save is not None:
        import os
        if not os.path.isdir(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)
        file_to_save = os.path.join(PLOTS_DIR, file_to_save)
        plt.savefig(filename=file_to_save)
    else:
        plt.show(block=block)

# custom loss function to mask out "black" area of images
# seems like there should be a less-hacky way to do this, but it will work for now
mask = K.variable(np.repeat(np.load('/project/mask.npy').flatten(), 16*334))

def eeg_mse(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    batch_size = K.prod(K.shape(y_true))
    y_pred *= mask[:batch_size]
    y_true *= mask[:batch_size]
    return objectives.mean_squared_error(y_true, y_pred)