import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_events(events_file) -> pd.DataFrame:
    """Load a BIDS-formatted events TSV file."""
    return pd.read_csv(events_file, sep="\t")


def filter_condition(events_df, start_marker="B3", end_marker="End B3") -> pd.DataFrame:
    """Keep only events between start_marker and end_marker (inclusive)."""
    start = events_df[events_df["value"] == start_marker]["onset"].values[0]
    end   = events_df[events_df["value"] == end_marker]["onset"].values[0]
    return events_df[
        (events_df["onset"] >= start) & (events_df["onset"] <= end)
    ].reset_index(drop=True)


def extract_rhs_cycles(events_df) -> list[tuple[float, float]]:
    """
    Return a list of (start_sec, end_sec) tuples from consecutive RHS events.
    Onsets are kept in seconds — convert to samples in the calling script.
    """
    rhs = events_df[events_df["value"] == "RHS"]["onset"].to_numpy(dtype=float)
    if len(rhs) < 2:
        raise ValueError(f"Need at least 2 RHS events to form a cycle; found {len(rhs)}.")
    return [(rhs[i], rhs[i + 1]) for i in range(len(rhs) - 1)]


def get_toe_off_onsets(events_df) -> np.ndarray | None:
    """
    Return RTO (right toe-off) onset times in seconds, or None if not present.
    Used for two-phase normalization.
    """
    if "RTO" not in events_df["value"].values:
        return None
    return events_df[events_df["value"] == "RTO"]["onset"].to_numpy(dtype=float)


def time_normalize(data: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resample a (n_channels, n_samples) array to (n_channels, n_points)
    using linear interpolation.
    """
    t_orig = np.linspace(0, 1, data.shape[1])
    t_new  = np.linspace(0, 1, n_points)
    out = np.zeros((data.shape[0], n_points), dtype=np.float32)
    for ch in range(data.shape[0]):
        out[ch] = interp1d(t_orig, data[ch], kind="linear")(t_new)
    return out


def extract_cycle_simple(raw, start_sec, end_sec, sfreq, n_points) -> np.ndarray:
    """
    Extract one gait cycle and time-normalize it to n_points.
    Returns array of shape (n_channels, n_points).
    """
    start_samp = int(np.round(start_sec * sfreq))
    end_samp   = int(np.round(end_sec   * sfreq))
    cycle_data = raw.get_data(start=start_samp, stop=end_samp)
    return time_normalize(cycle_data, n_points)


def extract_cycle_twophase(
    raw, start_sec, end_sec, toe_off_sec, sfreq, n_stance, n_swing
) -> np.ndarray | None:
    """
    Extract one gait cycle and normalize stance and swing phases separately.
    Returns array of shape (n_channels, n_stance + n_swing), or None if either
    phase is too short to interpolate.
    """
    start_samp   = int(np.round(start_sec   * sfreq))
    toe_off_samp = int(np.round(toe_off_sec * sfreq))
    end_samp     = int(np.round(end_sec     * sfreq))

    stance_data = raw.get_data(start=start_samp,   stop=toe_off_samp)
    swing_data  = raw.get_data(start=toe_off_samp, stop=end_samp)

    if stance_data.shape[1] < 2 or swing_data.shape[1] < 2:
        return None

    stance_norm = time_normalize(stance_data, n_stance)
    swing_norm  = time_normalize(swing_data,  n_swing)
    return np.concatenate([stance_norm, swing_norm], axis=1)


def print_cycle_qc(durations: list[float]) -> None:
    """Print a simple QC summary of cycle durations."""
    d = np.array(durations)
    print(f"  Duration: {d.mean():.3f} ± {d.std():.3f} s  [{d.min():.3f}, {d.max():.3f}]")