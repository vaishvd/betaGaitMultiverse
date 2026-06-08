"""
utils/beta_gait.py
==================
Utility functions for beta-band gait-cycle ERSP analysis.

Operates exclusively on ERSP arrays already produced by
ana05_gaitcycles2tfr.py.  No CSD transform is applied here — input data
is assumed to be ICA-cleaned only (no CSD influence).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# ─── constants matching ana05 ─────────────────────────────────────────────────

_FREQS_DEFAULT = np.arange(13, 31, dtype=float)   # 18 beta frequencies
_N_POINTS      = 101  # matches N_POINTS in ana05 (0-100% gait cycle, 1% steps)
_EDGE_CROP     = 0.05                              # fraction cropped at each edge


# ─── 1. load_tfr ─────────────────────────────────────────────────────────────

def load_tfr(ersp_path):
    """
    Load an ERSP .npy file produced by ana05.

    Parameters
    ----------
    ersp_path : str | Path
        Path to *_ersp_beta.npy  shape (n_ch, n_freqs, n_time).

    Returns
    -------
    ersp : ndarray (n_ch, n_freqs, n_time)

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    ValueError         if the array is not 3-D.
    """
    ersp_path = Path(ersp_path)
    if not ersp_path.exists():
        raise FileNotFoundError(f"ERSP file not found: {ersp_path}")

    ersp = np.load(ersp_path)

    if ersp.ndim != 3:
        raise ValueError(
            f"Expected 3-D ERSP (n_ch × n_freqs × n_time), got shape {ersp.shape}"
        )

    print(f"  load_tfr: shape={ersp.shape}  range=[{ersp.min():.2f}, {ersp.max():.2f}] dB")
    return ersp


# ─── 2. extract_beta_band ────────────────────────────────────────────────────

def extract_beta_band(ersp, freqs, f_low=13.0, f_high=30.0):
    """
    Clip ersp and freqs to the beta band [f_low, f_high] Hz.

    If the requested band extends outside the available frequencies it is
    clipped safely to what is available (no error is raised).

    Parameters
    ----------
    ersp  : ndarray (n_ch, n_freqs, n_time) or (n_freqs, n_time)
    freqs : ndarray (n_freqs,)
    f_low, f_high : float  — Hz boundaries, inclusive

    Returns
    -------
    beta_ersp  : ndarray — same ndim as input, freq axis restricted
    beta_freqs : ndarray (n_beta_freqs,)
    """
    freqs = np.asarray(freqs, dtype=float)
    mask  = (freqs >= f_low) & (freqs <= f_high)

    if not mask.any():
        # Requested band fully outside available range — return unchanged
        print(
            f"  extract_beta_band: requested [{f_low}, {f_high}] Hz is outside "
            f"available [{freqs.min()}, {freqs.max()}] Hz — returning full array."
        )
        return ersp, freqs

    beta_freqs = freqs[mask]

    if ersp.ndim == 3:
        beta_ersp = ersp[:, mask, :]
    elif ersp.ndim == 2:
        beta_ersp = ersp[mask, :]
    else:
        raise ValueError(f"ersp must be 2-D or 3-D, got {ersp.ndim}-D")

    return beta_ersp, beta_freqs


# ─── 3. normalize_to_gait_percent ────────────────────────────────────────────

def normalize_to_gait_percent(n_time, n_points=_N_POINTS, edge_crop=_EDGE_CROP):
    """
    Map n_time retained samples to gait-cycle percentage (0–100 %).

    The mapping accounts for the edge crop applied in ana05:
    e.g.  n_points=512, edge_crop=0.05 → crop=25 → n_time=462 → 5 %–95 %.

    Parameters
    ----------
    n_time     : int   — number of retained samples (post-crop)
    n_points   : int   — original samples per cycle (pre-crop, default 512)
    edge_crop  : float — fraction cropped at each edge (default 0.05)

    Returns
    -------
    pct : ndarray (n_time,)  in range [edge_crop*100, (1-edge_crop)*100]
    """
    crop      = int(edge_crop * n_points)
    pct_start = crop / n_points * 100.0
    pct_end   = (n_points - crop) / n_points * 100.0
    pct       = np.linspace(pct_start, pct_end, n_time)

    # Validation
    assert len(pct) == n_time, "pct length mismatch"
    return pct


# ─── 4. compute_beta_ersp ────────────────────────────────────────────────────

def compute_beta_ersp(
    ersp,
    freqs=None,
    ch_indices=None,
    f_low=13.0,
    f_high=30.0,
):
    """
    Compute channel-averaged, beta-band ERSP.

    Parameters
    ----------
    ersp       : ndarray (n_ch, n_freqs, n_time)
    freqs      : ndarray (n_freqs,) | None — if None, uses np.arange(13, 31)
    ch_indices : list of int | None — channels to average; None = all
    f_low, f_high : float — Hz boundaries for beta band

    Returns
    -------
    beta_avg   : ndarray (n_time,)         — freq- and channel-averaged ERSP
    beta_2d    : ndarray (n_beta_freqs, n_time) — channel-averaged, freq-resolved
    beta_freqs : ndarray (n_beta_freqs,)
    """
    if ersp.ndim != 3:
        raise ValueError(
            f"compute_beta_ersp expects (n_ch, n_freqs, n_time), got {ersp.shape}"
        )

    if freqs is None:
        n_freqs = ersp.shape[1]
        freqs   = _FREQS_DEFAULT[:n_freqs]
        if len(freqs) != n_freqs:
            freqs = np.arange(f_low, f_low + n_freqs, dtype=float)

    # Select channels
    data     = ersp[ch_indices] if ch_indices is not None else ersp
    ch_mean  = data.mean(axis=0)                         # (n_freqs, n_time)

    # Extract beta band
    beta_2d, beta_freqs = extract_beta_band(ch_mean, freqs, f_low, f_high)
    beta_avg = beta_2d.mean(axis=0)                      # (n_time,)

    print(
        f"  compute_beta_ersp: {len(data)} ch  "
        f"{len(beta_freqs)} freqs ({beta_freqs[0]:.0f}–{beta_freqs[-1]:.0f} Hz)  "
        f"→ beta_avg shape={beta_avg.shape}"
    )
    return beta_avg, beta_2d, beta_freqs


# ─── 5. plot_beta_gait_heatmap ───────────────────────────────────────────────

def plot_beta_gait_heatmap(
    beta_2d,
    pct_axis,
    freqs,
    out_path,
    event_pcts=None,
    title="",
    smooth_sigma=0,
    collapse_freq=True,
):
    """
    Publication-quality beta-gait heatmap.

    Parameters
    ----------
    beta_2d      : ndarray (n_freqs, n_time) — channel-averaged, beta-band ERSP
    pct_axis     : ndarray (n_time,)  gait-cycle % values
    freqs        : ndarray (n_freqs,) Hz
    out_path     : str | Path — figure save location
    event_pcts   : dict {label: pct_value} | None  — gait events to mark
    title        : str  — figure suptitle (empty = omit)
    smooth_sigma : int  — Gaussian σ applied along time axis (0 = off)
    collapse_freq: bool — True  → single-row heatmap (freq-averaged, preferred)
                          False → full (n_freqs × n_time) frequency heatmap

    Saved figure labels
    -------------------
    x-axis : "Gait Cycle (%)"
    colorbar label : "Beta Power (13–30 Hz, dB)"
    """
    # Input validation
    if beta_2d.ndim != 2:
        raise ValueError(f"beta_2d must be 2-D (n_freqs × n_time), got {beta_2d.shape}")
    if beta_2d.shape[-1] != len(pct_axis):
        raise ValueError(
            f"beta_2d time axis ({beta_2d.shape[-1]}) != pct_axis length ({len(pct_axis)})"
        )

    data = beta_2d.copy()
    if smooth_sigma > 0:
        data = gaussian_filter1d(data, sigma=smooth_sigma, axis=-1)

    ev_styles = {
        "RHS": ("white",  "solid",  1.5),
        "LTO": ("black",  "dashed", 1.2),
        "LHS": ("black",  "solid",  1.2),
        "RTO": ("black",  "dotted", 1.2),
    }

    def _add_event_lines(ax):
        if not event_pcts:
            return
        for ev, pv in event_pcts.items():
            if pct_axis[0] <= pv <= pct_axis[-1]:
                color, ls, lw = ev_styles.get(ev, ("gray", "solid", 1.0))
                ax.axvline(pv, color=color, ls=ls, lw=lw, label=ev)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels,
                loc="upper right", fontsize=7,
                framealpha=0.6, ncol=len(handles),
            )

    if collapse_freq:
        # ── Single-row heatmap (freq-averaged) ────────────────────────────
        row  = data.mean(axis=0, keepdims=True)    # (1, n_time)
        vmax = np.abs(row).max()
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(10, 1.8))

        im = ax.imshow(
            row,
            aspect="auto",
            origin="lower",
            extent=[pct_axis[0], pct_axis[-1], 0, 1],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_yticks([])
        ax.set_ylabel("Beta\n(13–30 Hz)", fontsize=9)
        ax.set_xlabel("Gait Cycle (%)", fontsize=10)
        ax.set_xticks([0, 25, 50, 75, 100])

        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label("Beta Power (13–30 Hz, dB)", fontsize=9)

        _add_event_lines(ax)

    else:
        # ── Full frequency × time heatmap ─────────────────────────────────
        vmax = np.abs(data).max()
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(10, 4))

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[pct_axis[0], pct_axis[-1], freqs[0] - 0.5, freqs[-1] + 0.5],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_ylabel("Frequency (Hz)", fontsize=10)
        ax.set_xlabel("Gait Cycle (%)", fontsize=10)
        ax.set_yticks(freqs[::2])
        ax.set_xticks([0, 25, 50, 75, 100])

        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Beta Power (13–30 Hz, dB)", fontsize=9)

        _add_event_lines(ax)

    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
