import numpy as np
import mne


def compute_tfr(data: np.ndarray, sfreq: float, freqs: np.ndarray) -> np.ndarray:
    """
    Compute Morlet wavelet power for gait-cycle EEG data.

    Parameters
    ----------
    data   : (n_cycles, n_channels, n_timepoints)
    sfreq  : sampling frequency in Hz
    freqs  : frequency array, e.g. np.arange(13, 31)

    Returns
    -------
    power  : (n_cycles, n_channels, n_freqs, n_timepoints)

    Notes
    -----
    n_cycles is set to freqs / 2 (half-wavelength rule) but capped so that
    no wavelet is longer than the signal. This is necessary when cycles are
    time-normalized to a fixed number of points (e.g. 200), which can be
    shorter than the wavelet at low frequencies.

    The cap formula is: max_cycles = (n_timepoints - 1) / 2 / sfreq * freqs
    which is the largest n_cycles that keeps the wavelet within the signal.
    """
    n_timepoints = data.shape[2]

    n_cycles_wanted = freqs / 2
    n_cycles_max    = (n_timepoints - 1) / 2.0 / sfreq * freqs
    n_cycles        = np.minimum(n_cycles_wanted, n_cycles_max)

    print(f"  Wavelet n_cycles: min={n_cycles.min():.2f}, max={n_cycles.max():.2f} "
          f"(signal length: {n_timepoints} samples)")

    power = mne.time_frequency.tfr_array_morlet(
        data,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
    )
    return power  # (n_cycles, n_channels, n_freqs, n_timepoints)