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
 
        i0, i1 = int(start * sfreq), int(end * sfreq)
        if i0 < 0 or i1 > raw.n_times:
            continue
 
        data = raw.get_data(start=i0, stop=i1)
        if data.shape[1] < 10:
            continue
 
        segments.append(data)
        durations.append(dur)
        p2p.append(float(np.median(np.ptp(data, axis=1))))
 
    if not segments:
        return None, None
 
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


# Sanity check: plot raw cycles before any rejection
 
def plot_cycle_diagnostics(cycles: list[tuple[float, float]], raw, path) -> None:
    """
    Overlay first 20 raw cycles (channel 0) before any rejection —
    verifies event-to-EEG alignment independently of artefact status.
    """
    sfreq = raw.info["sfreq"]
    fig, ax = plt.subplots(figsize=(6, 4))
 
    for start, end in cycles[:20]:
        data = raw.get_data(start=int(start * sfreq), stop=int(end * sfreq))
        if data.shape[1] >= 10:
            ax.plot(np.linspace(0, 100, data.shape[1]), data[0] * 1e6, alpha=0.2, lw=0.8)
 
    ax.set(xlabel="Gait cycle (%)", ylabel="Amplitude (µV)",
           title="Raw cycle alignment — first 20 cycles (ch 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)