"""
ana06_plotbetagait.py
=====================
Plot beta-band (13-30 Hz) ERSP across the normalised gait cycle.

Input
-----
d05_ersp/       sub-{sub}_ersp_beta.npy        (n_ch x n_freqs x 101)
d04_gaitepochs/ sub-{sub}_cycles_kept.tsv      (for mean gait event positions)
d03_clean/      sub-{sub}_desc-icaClean_concat_raw.fif  (for channel names)

Output
------
results/plots/  group_beta_ersp_gait_n{n}.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
from pathlib import Path

from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS, DIR_PLOTS, DIR_QC
from src.qc import log_qc

ROI_CHANNEL  = ["Cz","C3","C4", "FCz", "FC3", "FC4"]  # central midline + nearby channels; adjust if missing
FREQS        = np.arange(13, 31)
ERSP_CLIM    = 4.0        # symmetric dB limit; adjust if group mean is clipped
STANCE_COLOR = "#DDEEFF"  # light blue
SWING_COLOR  = "#FFF3DD"  # light orange

dirs          = get_dataset_dirs(DATASET)
ERSP_DIR      = dirs["ersp"]
CLEAN_DIR     = dirs["clean"]
GAITEPOCH_DIR = dirs["gaitepochs"]
PLOTS_DIR     = Path(DIR_PLOTS)
QC_DIR        = Path(DIR_QC)
ersp_list   = []
events_list = []

for subject in SUBJECTS:
    try:
        ersp_path   = ERSP_DIR      / f"sub-{subject}_ersp_beta.npy"
        cycles_path = GAITEPOCH_DIR / f"sub-{subject}_cycles_kept.tsv"
        clean_path  = CLEAN_DIR     / f"sub-{subject}_desc-icaClean_concat_raw.fif"

        ersp     = np.load(ersp_path)   # (n_ch, n_freqs, 101)
        cycles   = pd.read_csv(cycles_path, sep="\t")
        raw_ref  = mne.io.read_raw_fif(clean_path, preload=False, verbose=False)
        ch_names = list(raw_ref.ch_names)

        if not any(channel in ch_names for channel in ROI_CHANNEL):
            print(f"  [WARN] sub-{subject}: None of {ROI_CHANNEL} in channel list -- skipping.")
            continue

        roi_indices = [ch_names.index(channel) for channel in ROI_CHANNEL if channel in ch_names]
        ersp_roi = np.mean(ersp[roi_indices], axis=0)  # (n_freqs, 101)

        dur      = cycles["rhs_end_s"].values - cycles["rhs_start_s"].values
        lto_pct  = (cycles["lto_s"].values  - cycles["rhs_start_s"].values) / dur * 100
        lhs_pct  = (cycles["lhs_s"].values  - cycles["rhs_start_s"].values) / dur * 100
        rto_pct  = (cycles["rto_s"].values  - cycles["rhs_start_s"].values) / dur * 100
        mean_lto = float(np.mean(lto_pct))
        mean_lhs = float(np.mean(lhs_pct))
        mean_rto = float(np.mean(rto_pct))

        ersp_list.append(ersp_roi)
        events_list.append((mean_lto, mean_lhs, mean_rto))

        print(f"  sub-{subject}: {ROI_CHANNEL} mean={ersp_roi.mean():+.2f} dB  "
              f"LTO={mean_lto:.1f}%  LHS={mean_lhs:.1f}%  RTO={mean_rto:.1f}%")

    except FileNotFoundError as e:
        print(f"\n  [SKIP] sub-{subject}: file not found -- {e}")
        continue

if len(ersp_list) == 0:
    print("No subjects loaded. Exiting.")
    raise SystemExit(1)

# Group average
ersp_group = np.mean(np.stack(ersp_list), axis=0)  # (n_freqs, 101)
beta_trace = ersp_group.mean(axis=0)                # (101,) mean across freqs

# Group mean gait events
mean_lto_group = float(np.mean([e[0] for e in events_list]))
mean_lhs_group = float(np.mean([e[1] for e in events_list]))
mean_rto_group = float(np.mean([e[2] for e in events_list]))

n_subjects = len(ersp_list)
gait_pct   = np.linspace(0, 100, ersp_group.shape[1])

print(f"\n  Group average: n={n_subjects} subjects")
print(f"  Group ERSP range: {ersp_group.min():.2f} / {ersp_group.max():.2f} dB")
print(f"  Group events: LTO={mean_lto_group:.1f}%  "
      f"LHS={mean_lhs_group:.1f}%  RTO={mean_rto_group:.1f}%")

# Build group figure 

fig = plt.figure(figsize=(11, 8))
gs  = gridspec.GridSpec(
    2, 1,
    height_ratios=[3, 2],
    hspace=0.08
)
ax_heat  = fig.add_subplot(gs[0])
ax_trace = fig.add_subplot(gs[1], sharex=ax_heat)

fig.suptitle(
    f"Beta ERSP over Gait Cycle — Group Average  "
    f"(n={n_subjects}, {ROI_CHANNEL})",
    fontsize=12, fontweight="bold", y=0.98
)

# Panel A -- heatmap with dotted event lines
ax_heat.set_facecolor("white")

im = ax_heat.imshow(
    ersp_group,
    aspect="auto",
    origin="lower",
    extent=[0, 100, FREQS[0] - 0.5, FREQS[-1] + 0.5],
    cmap="RdBu_r",
    vmin=-ERSP_CLIM,
    vmax=+ERSP_CLIM,
    zorder=1,
)

# Gait event lines on heatmap 
heatmap_events = [
    (0,               "RHS"),
    (mean_lto_group,  "LTO"),
    (mean_lhs_group,  "LHS"),
    (mean_rto_group,  "RTO"),
    (100,             "RHS"),
]
for pct, label in heatmap_events:
    ax_heat.axvline(
        pct, color="black", linewidth=1.0,
        linestyle=":", zorder=3
    )

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
cbar.set_label("ERSP (dB)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax_heat.set_ylabel("Frequency (Hz)", fontsize=10)
ax_heat.set_ylim(FREQS[0] - 0.5, FREQS[-1] + 0.5)
ax_heat.set_yticks([13, 16, 20, 24, 28, 30])
ax_heat.tick_params(labelsize=8, labelbottom=False)

# Panel C -- beta trace with stance/swing shading and event labels
# Stance/swing shading
ax_trace.axvspan(0,              mean_rto_group, color=STANCE_COLOR,
                 alpha=0.45, zorder=0, label="Stance")
ax_trace.axvspan(mean_rto_group, 100,            color=SWING_COLOR,
                 alpha=0.45, zorder=0, label="Swing")

# Gait event dotted lines -- black, matching heatmap
trace_events = [
    (0,               "RHS"),
    (mean_lto_group,  "LTO"),
    (mean_lhs_group,  "LHS"),
    (mean_rto_group,  "RTO"),
    (100,             "RHS"),
]
for pct, label in trace_events:
    ax_trace.axvline(
        pct, color="black", linewidth=1.0,
        linestyle=":", zorder=3
    )

# Event labels just above the x-axis inside Panel C
for pct, label in trace_events:
    ax_trace.text(
        pct, 1.02, label,
        transform=ax_trace.get_xaxis_transform(),
        ha="center", va="bottom",
        fontsize=7, color="black",
        fontweight="bold"
    )

ax_trace.axhline(0, color="black", linewidth=0.8,
                 linestyle=":", zorder=1)
ax_trace.plot(
    gait_pct, beta_trace,
    color="#1f3d7a", linewidth=2.0,
    zorder=2, label=f"{', '.join(ROI_CHANNEL)}  n={n_subjects}"
)
ax_trace.set_xlabel("Gait cycle (%)", fontsize=10)
ax_trace.set_ylabel("ERSP (dB)", fontsize=10)
ax_trace.set_xlim(0, 100)
ax_trace.tick_params(labelsize=8)
ax_trace.legend(loc="lower left", fontsize=8, framealpha=0.7)

out_path = PLOTS_DIR / f"group_beta_ersp_gait_n{n_subjects}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Group figure saved -> {out_path.name}")
