import numpy as np
 
 
def log_power(power: np.ndarray) -> np.ndarray:
    """Convert power to decibels (10 * log10). Input: (n_cycles, n_channels, n_freqs, n_timepoints)."""
    return 10 * np.log10(power)
 
 
def baseline_correct(
    power_db: np.ndarray,
    baseline_pct: tuple[float, float] = (0.0, 0.1),
) -> np.ndarray:
    """
    Subtract a baseline from log-power, averaged across cycles.
 
    The baseline is the mean over a window defined as a fraction of the
    normalized gait cycle (0.0–1.0). The default (0.0, 0.1) uses the first
    10% of the cycle as baseline — a common choice when no separate rest
    condition is available.
 
    For a proper pre-movement baseline, record a standing-still condition
    and pass those cycles here instead.
 
    Parameters
    ----------
    power_db     : (n_cycles, n_channels, n_freqs, n_timepoints) in dB
    baseline_pct : (start, end) as fraction of the cycle, e.g. (0.0, 0.1)
 
    Returns
    -------
    ersp : (n_cycles, n_channels, n_freqs, n_timepoints)
           baseline-corrected, averaged across cycles
    """
    n_timepoints = power_db.shape[-1]
    t_start = int(np.round(baseline_pct[0] * n_timepoints))
    t_end   = int(np.round(baseline_pct[1] * n_timepoints))
 
    # Average baseline across the window and across all cycles
    # shape after mean: (1, n_channels, n_freqs, 1) — broadcasts cleanly
    baseline = power_db[:, :, :, t_start:t_end].mean(axis=(0, 3), keepdims=True)
 
    ersp = power_db - baseline
    return ersp  # (n_cycles, n_channels, n_freqs, n_timepoints)
 