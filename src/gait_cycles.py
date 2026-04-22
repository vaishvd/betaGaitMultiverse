import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_events(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


# ── Event parsing ─────────────────────────────────────────────────────────────

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


# ── Cycle extraction & rejection ──────────────────────────────────────────────

def extract_cycles(
    raw, cycles: list[tuple[float, float]],
    sfreq: float, min_dur: float, max_dur: float,
    n_points: int, k: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Two-pass pipeline:
      Pass 1 — collect valid segments and compute per-cycle median P2P.
      Pass 2 — reject cycles above median + k*MAD, time-normalise survivors.

    Exclusion criteria (pass 1):
      - duration outside [min_dur, max_dur]
      - segment outside recording bounds
      - fewer than 10 samples (degenerate)

    Returns:
        cycles_out : (n_cycles, channels, n_points)  or None
        durations  : (n_cycles,)                     or None
    """
    # ── Pass 1: collect ───────────────────────────────────────────────────────
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

    # ── Threshold: median + k * MAD ───────────────────────────────────────────
    p2p_arr   = np.array(p2p)
    med       = np.median(p2p_arr)
    threshold = med + k * np.median(np.abs(p2p_arr - med))

    # ── Pass 2: reject + time-normalise ──────────────────────────────────────
    x_new = np.linspace(0, 1, n_points)

    clean, clean_dur = [], []
    for data, dur, amp in zip(segments, durations, p2p):
        if amp > threshold:
            continue
        x_old = np.linspace(0, 1, data.shape[1])
        clean.append(np.array([np.interp(x_new, x_old, ch) for ch in data]))
        clean_dur.append(dur)

    if not clean:
        return None, None

    return np.stack(clean), np.array(clean_dur)


# ── Diagnostics ───────────────────────────────────────────────────────────────

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


def plot_p2p_distribution(p2p_values: list[float], threshold: float, path) -> None:
    """Histogram of per-cycle median P2P with rejection threshold marked."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(np.array(p2p_values) * 1e6, bins=30)
    ax.axvline(threshold * 1e6, color="r", ls="--",
               label=f"Threshold  {threshold * 1e6:.0f} µV")
    ax.set(xlabel="Cycle median P2P (µV)", ylabel="Count", title="P2P distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)