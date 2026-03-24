import numpy as np
import mne


def _pad_cycles(data: np.ndarray, pad: int) -> np.ndarray:
    """
    Reflect-pad each cycle at both ends to suppress edge artefacts.

    Mirrors the first and last `pad` timepoints so the wavelet has valid
    data to convolve with at the cycle boundaries. The pad is cropped off
    after TFR computation.

    Parameters
    ----------
    data : (n_cycles, n_channels, n_timepoints)
    pad  : number of timepoints to mirror at each end

    Returns
    -------
    padded : (n_cycles, n_channels, n_timepoints + 2*pad)
    """
    left  = data[:, :, :pad][:, :, ::-1]   # mirror of first `pad` points
    right = data[:, :, -pad:][:, :, ::-1]  # mirror of last `pad` points
    return np.concatenate([left, data, right], axis=2)


def compute_tfr(data: np.ndarray, sfreq: float, freqs: np.ndarray) -> np.ndarray:
    """
    Compute Morlet wavelet power for gait-cycle EEG data, with reflection
    padding to suppress edge artefacts.

    Parameters
    ----------
    data   : (n_cycles, n_channels, n_timepoints)
    sfreq  : sampling frequency in Hz
    freqs  : frequency array, e.g. np.arange(13, 31)

    Returns
    -------
    power  : (n_cycles, n_channels, n_freqs, n_timepoints)
             cropped back to original n_timepoints after padding

    Notes
    -----
    Reflection padding: each cycle is mirrored at both ends by `pad` samples
    before TFR, then cropped. This prevents the wavelet from running off the
    edge of the signal and producing the characteristic blue artefact at 0%
    and 100% of the gait cycle.

    The pad length is set to half the longest wavelet in the bank, which is
    the minimum needed to fully cover the edge for every frequency.
    """
    n_timepoints = data.shape[2]

    n_cycles_wanted = freqs / 2
    n_cycles_max    = (n_timepoints - 1) / 2.0 / sfreq * freqs
    n_cycles        = np.minimum(n_cycles_wanted, n_cycles_max)

    # Pad length = half the longest wavelet (in samples), rounded up
    longest_wavelet_samples = int(np.ceil(n_cycles.max() / freqs.min() * sfreq))
    pad = longest_wavelet_samples // 2

    data_padded = _pad_cycles(data, pad)

    print(f"  Wavelet n_cycles: min={n_cycles.min():.2f}, max={n_cycles.max():.2f} "
          f"| pad: {pad} samples each side")

    power_padded = mne.time_frequency.tfr_array_morlet(
        data_padded,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
    )

    # Crop padding back out → original n_timepoints
    power = power_padded[:, :, :, pad: pad + n_timepoints]
    return power  # (n_cycles, n_channels, n_freqs, n_timepoints)