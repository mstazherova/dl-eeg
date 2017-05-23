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


def raw_to_image(raw_data, locs_3d, sfreq, n_gridpoints=32):
    n_channels = raw_data.shape[0]
    sample_rate = 1 / sfreq
    channels_samples = []
    n_windows = 10
    for channel_data in raw_data[:n_channels]:
        samples = []
        window_len = int(channel_data.shape[0] / n_windows)
        for window_idx in range(n_windows):
            start_idx = max(0, window_idx * window_len - int(window_len / 2))
            end_idx = (window_idx + 1) * window_len
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
        normalize=False,
    )

    return images
