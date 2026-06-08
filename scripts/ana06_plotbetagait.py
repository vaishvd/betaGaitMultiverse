"""
ana06_plotbetagait.py
=====================
Plot beta-band (13-30 Hz) ERSP across the normalised gait cycle.

Input
-----
d05_ersp/       sub-{sub}_ersp_beta.npy   (n_ch x n_freqs x n_time)
d01_gaitevents/ sub-{sub}_cycles.tsv      (for mean event positions)
d03_clean/      sub-{sub}_desc-icaClean_raw.fif  (for channel names)

Output
------
results/plots/  sub-{sub}_sensorimotor_beta_ersp.png
"""

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.paths import get_dataset_dirs
from src.config import DIR_PLOTS

DATASET  = "stepup"
SUBJECTS = ["S1"]
CHANNEL = ["Cz", "C3", "C4","FC1", "FC2", "FCz", "CP1", "CP2", "CPz"]  # channels to plot

FREQS = np.arange(13, 31, dtype=float)

dirs            = get_dataset_dirs(DATASET)
ERSP_DIR        = dirs["ersp"]
CLEAN_DIR       = dirs["clean"]
GAIT_EVENTS_DIR = dirs["gait_events"]
PLOTS_DIR       = DIR_PLOTS

for subject in SUBJECTS:

    print(f"\n{subject} -- Plotting beta ERSP")

    ersp_path = ERSP_DIR / f"sub-{subject}_ersp_beta.npy"
    if not ersp_path.exists():
        print(f"  ERSP file not found: {ersp_path.name} -- skipping.")
        continue

    ersp = np.load(ersp_path)    # (n_ch, n_freqs, n_time)

    raw_ref = mne.io.read_raw_fif(
        CLEAN_DIR / f"sub-{subject}_desc-icaClean_raw.fif",
        preload=False, verbose=False,
    )
    ch_names = list(raw_ref.ch_names)

    cycles_df = pd.read_csv(
        GAIT_EVENTS_DIR / f"sub-{subject}_cycles.tsv", sep="\t"
    )
    span = cycles_df["rhs_end"] - cycles_df["rhs_start"]
    event_pcts = {
        "RHS": 0.0,
        "LTO": cycles_df["lto_frac"].mean() * 100,
        "LHS": ((cycles_df["lhs"] - cycles_df["rhs_start"]) / span).mean() * 100,
        "RTO": ((cycles_df["rto"] - cycles_df["rhs_start"]) / span).mean() * 100,
    }

    ch_idx = [ch_names.index(ch) for ch in CHANNEL if ch in ch_names]
    # Average across channels
    ersp_roi = ersp[ch_idx].mean(axis=0)   # (n_freqs, n_time)

    vmax = np.max(np.abs(ersp_roi))

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Heatmap

    im = ax1.imshow(
        ersp_roi,
        aspect="auto",
        origin="lower",
        extent=[0, 100, FREQS[0], FREQS[-1]],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax1, label="ERSP (dB)")

    for ev, x in event_pcts.items():
        ax1.axvline(x, ls="--", color="k", lw=1)

    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(
        f"sub-{subject} Sensorimotor ERSP"
    )

    # Mean beta trace

    beta_trace = ersp_roi.mean(axis=0)

    ax2.plot(
        np.linspace(0, 100, len(beta_trace)),
        beta_trace,
        lw=2,
    )

    for ev, x in event_pcts.items():
        ax2.axvline(x, ls="--", color="k", lw=1)

    ax2.set_xlabel("Gait cycle (%)")
    ax2.set_ylabel("Beta ERSP (dB)")

    plt.tight_layout()

    out = PLOTS_DIR / f"sub-{subject}_sensorimotor_beta_ersp.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

print("\nPlotting done")
