from .eeglearn.eeg_cnn_lib import gen_images, azim_proj
import numpy as np

FREQ_RANGES = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
}


def sample_to_channels(sample, freqs):
    """
    :param sample: EEG time series after applying FFT
    :param freqs: list of frequencies
    :return:
    """
    assert sample.shape[0] == freqs.shape[0]
    assert len(sample) == freqs.shape[0]

    theta, alpha, beta = [], [], []
    for freq, val in zip(freqs, sample):
        freq_band = freq_to_band(freq)
        if freq_band == 'theta':
            theta.append(val)
        elif freq_band == 'alpha':
            alpha.append(val)
        elif freq_band == 'beta':
            beta.append(val)

    theta, alpha, beta = np.array(theta), np.array(alpha), np.array(beta)
    # compute sum of squared elements for each frequency band
    theta, alpha, beta = np.sum(theta ** 2), np.sum(alpha ** 2), np.sum(beta ** 2)
    return theta, alpha, beta


def freq_to_band(frequency):
    for freq_name, freq_range in FREQ_RANGES.items():
        if freq_range[0] <= frequency <= freq_range[1]:
            return freq_name
    return None


def raw_to_image(raw_data, locs_3d, sfreq, window_len=0.5, single_frame=False, n_gridpoints=32, normalize=True):
    n_channels = raw_data.shape[0]
    sample_rate = 1 / sfreq
    channels_samples = []

    if single_frame:
        # perform FFT on the whole series (thus leading to only one image)
        n_windows = 1
    else:
        # total number of windows is equal to the length of the series in seconds (which is number of entries / sfreq)
        # divided by the length of the windows (in seconds)
        n_windows = int(raw_data.shape[1] / sfreq / window_len)

    for channel_data in raw_data[:n_channels]:
        samples = []
        for window_idx in range(n_windows):
            if n_windows == 1:
                # single frame approach
                start_idx = 0
                end_idx = len(channel_data)
            else:
                start_idx = max(0, int((window_idx - 1) * window_len * sfreq))
                end_idx = int((window_idx + 1) * window_len * sfreq)

            window_data = channel_data[start_idx:end_idx]
            fft = np.fft.fft(window_data)
            freqs = np.fft.fftfreq(len(fft), sample_rate)
            theta, alpha, beta = sample_to_channels(
                sample=fft,
                freqs=freqs
            )
            samples.append([theta, alpha, beta])
        channels_samples.append(np.array(samples))

    feats = None
    for band_idx in range(3):
        for samples in channels_samples:
            if feats is None:
                feats = samples[:, band_idx].reshape(samples.shape[0], 1)
            else:
                feats = np.concatenate((feats, samples[:, band_idx].reshape(samples.shape[0], 1)), axis=1)

    locs_2d = []
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    images = gen_images(
        locs=np.array(locs_2d),
        features=feats,
        n_gridpoints=n_gridpoints,
        normalize=normalize,
    )

    return images
