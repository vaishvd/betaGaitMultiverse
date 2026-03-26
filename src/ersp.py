import numpy as np


def log_power(power: np.ndarray) -> np.ndarray:
    """Convert power to decibels (10 * log10). Input: (n_cycles, n_channels, n_freqs, n_timepoints)."""
    return 10 * np.log10(power)


def baseline_correct(power_db: np.ndarray) -> np.ndarray:
    """
    Baseline-correct log-power by subtracting the mean across ALL timepoints
    and ALL cycles, per channel and frequency.

    This is the standard approach when no separate rest condition is available.
    It is equivalent to what EEGLAB's newtimef uses as its default baseline and
    removes mean power level without assuming any part of the gait cycle is
    "neutral". The result expresses each time-frequency point as deviation from
    the cycle-averaged power at that frequency.

    Parameters
    ----------
    power_db : (n_cycles, n_channels, n_freqs, n_timepoints) in dB

    Returns
    -------
    ersp : (n_cycles, n_channels, n_freqs, n_timepoints)
    """
    # Mean over all cycles and all timepoints → (1, n_channels, n_freqs, 1)
    baseline = power_db.mean(axis=(0, 3), keepdims=True)
    return power_db - baseline

def baseline_per_cycle(tfr):
    # tfr shape: (n_cycles, n_channels, n_freqs, n_times)

    import numpy as np

    baseline = np.mean(tfr, axis=-1, keepdims=True)  # mean over time per cycle
    return 10 * np.log10(tfr / baseline)

def baseline_global(tfr):
    import numpy as np

    baseline = np.mean(tfr, axis=(0, -1), keepdims=True)
    return 10 * np.log10(tfr / baseline)