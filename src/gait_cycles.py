import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_events(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def filter_condition(events: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Trim events to the window [start_marker, end_marker]."""
    t0 = events.loc[events["value"] == start, "onset"].iat[0]
    t1 = events.loc[events["value"] == end,   "onset"].iat[0]
    return events[(events["onset"] >= t0) & (events["onset"] <= t1)].reset_index(drop=True)


def rhs_cycles(events: pd.DataFrame) -> list[tuple[float, float]]:
    """Return (start_s, end_s) pairs from consecutive RHS events — one pair = one stride."""
    onsets = events.loc[events["value"] == "RHS", "onset"].to_numpy(dtype=float)
    if len(onsets) < 2:
        raise ValueError(f"Need ≥2 RHS events; found {len(onsets)}.")
    return list(zip(onsets[:-1], onsets[1:]))


def compute_durations(cycles: list[tuple[float, float]]) -> np.ndarray:
    """Return stride durations in seconds."""
    return np.array([end - start for start, end in cycles])


# Cycle extraction & rejection

def extract_cycles(
    raw, cycles: list[tuple[float, float]],
    sfreq: float, min_dur: float, max_dur: float,
    k: float = 3.0,
) -> tuple[list | None, np.ndarray | None]:
    """
    Two-pass pipeline:
      Pass 1 — collect valid segments, compute per-cycle median P2P.
      Pass 2 — reject cycles above median + k*MAD.
 
    Returns:
        segments  : list of (channels, samples) arrays at true sfreq
        durations : (n_cycles,) stride durations in seconds
    """
    segments, durations, p2p = [], [], []
 
    for start, end in cycles:
        dur = end - start
        if not (min_dur <= dur <= max_dur):
            continue
 
    p2p_arr   = np.array(p2p)
    med       = np.median(p2p_arr)
    threshold = med + k * np.median(np.abs(p2p_arr - med))
 
    clean, clean_dur = zip(
        *[(data, dur) for data, dur, amp in zip(segments, durations, p2p) if amp <= threshold]
    ) if any(amp <= threshold for amp in p2p) else (None, None)
 
    if clean is None:
        return None, None
 
    return list(clean), np.array(clean_dur)

# Gait event structure in cycle space

def compute_event_means(
    events: pd.DataFrame, cycles: list[tuple[float, float]]
) -> dict[str, float | None]:
    """
    Return mean position of each gait event (RHS, LTO, LHS, RTO)
    as a percentage of the gait cycle, averaged across all cycles.
    """
    event_map: dict[str, list[float]] = {"RHS": [], "LTO": [], "LHS": [], "RTO": []}
 
    for start, end in cycles:
        cycle_events = events[(events["onset"] > start) & (events["onset"] < end)]
        for _, row in cycle_events.iterrows():
            ev = row["value"]
            if ev in event_map:
                event_map[ev].append((row["onset"] - start) / (end - start) * 100)
 
    return {ev: float(np.mean(vals)) if vals else None for ev, vals in event_map.items()}


# Plotting

def plot_duration_distribution(durations: np.ndarray, out_file):
    """Histogram of gait cycle durations."""
    plt.figure(figsize=(5, 3))

    plt.hist(durations, bins=30)
    plt.xlabel("Gait cycle duration (s)")
    plt.ylabel("Count")
    plt.title("Gait cycle duration distribution")

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()