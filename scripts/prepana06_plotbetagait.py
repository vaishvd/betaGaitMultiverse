"""
ana06_plotbetagait.py
=====================
Plot beta-band (13-30 Hz) ERSP across the normalised gait cycle.

Three panels side by side:
  1. Group beta ERSP heatmap with double stance / swing window lines
  2. Beta power topography (ERSP averaged over 13-30 Hz and full gait cycle)
  3. Spatial filter topography (linear ROI weights)

Input
-----
d05_ersp/       sub-{sub}_ersp_beta.npy        (n_ch x n_freqs x 101)
                sub-{sub}_roi_weights.npy       (n_ch,)
d04_gaitepochs/ sub-{sub}_cycles_kept.tsv      (for mean gait event positions)
d03_clean/      sub-{sub}_desc-icaClean_concat_raw.fif  (for channel info)

Output
------
results/pipeline/plots/  group_beta_ersp_topo.png
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
from src.config import DATASET, SUBJECTS, DIR_PLOTS
from src.spatial_filter import linear_roi_weights, apply_linear_roi

FREQS      = np.arange(13, 31)
ERSP_CLIM  = 4.0

# Literature-based peak windows (confirmatory, not data-driven)
# Double stance ERS at heel contact: Petersen et al. 2012 J Physiol
# Swing ERD: Bulea et al. 2015 Front Hum Neurosci;
#            Seeber et al. 2015 Front Hum Neurosci
DOUBLE_STANCE_WINDOWS = [(0, 20), (50, 70)]
SWING_WINDOWS         = [(20, 50), (70, 100)]

dirs          = get_dataset_dirs(DATASET)
ERSP_DIR      = dirs["ersp"]
CLEAN_DIR     = dirs["clean"]
GAITEPOCH_DIR = dirs["gaitepochs"]
PLOTS_DIR     = Path(DIR_PLOTS)

ersp_list    = []
topo_list    = []   # per-subject (n_ch,) ERSP averaged over freqs & time
weights_list = []   # per-subject (n_ch,) linear ROI weights
events_list  = []
info_ref     = None

for subject in SUBJECTS:
    try:
        ersp_path   = ERSP_DIR      / f"sub-{subject}_ersp_beta.npy"
        cycles_path = GAITEPOCH_DIR / f"sub-{subject}_cycles_kept.tsv"
        clean_path  = CLEAN_DIR     / f"sub-{subject}_desc-icaClean_concat_raw.fif"

        ersp     = np.load(ersp_path)   # (n_ch, n_freqs, 101)
        cycles   = pd.read_csv(cycles_path, sep="\t")
        raw_ref  = mne.io.read_raw_fif(clean_path, preload=False, verbose=False)

        # Load or compute linear ROI weights
        weights_path = ERSP_DIR / f"sub-{subject}_roi_weights.npy"
        if weights_path.exists():
            sub_weights = np.load(weights_path)
        else:
            sub_weights = linear_roi_weights(raw_ref.info, center_ch="Cz")

        # Apply weights: (n_ch, n_freqs, 101) → (n_freqs, 101)
        ersp_roi = apply_linear_roi(ersp, sub_weights)

        # For topography (Panel 2): average over freqs and full gait cycle
        ersp_topo = ersp.mean(axis=(1, 2))   # (n_ch,)

        dur      = cycles["rhs_end_s"].values - cycles["rhs_start_s"].values
        lto_pct  = (cycles["lto_s"].values  - cycles["rhs_start_s"].values) / dur * 100
        lhs_pct  = (cycles["lhs_s"].values  - cycles["rhs_start_s"].values) / dur * 100
        rto_pct  = (cycles["rto_s"].values  - cycles["rhs_start_s"].values) / dur * 100
        mean_lto = float(np.mean(lto_pct))
        mean_lhs = float(np.mean(lhs_pct))
        mean_rto = float(np.mean(rto_pct))

        ersp_list.append(ersp_roi)
        topo_list.append(ersp_topo)
        weights_list.append(sub_weights)
        events_list.append((mean_lto, mean_lhs, mean_rto))
        info_ref = raw_ref.info

        print(f"  sub-{subject}: Linear ROI mean={ersp_roi.mean():+.2f} dB  "
              f"LTO={mean_lto:.1f}%  LHS={mean_lhs:.1f}%  RTO={mean_rto:.1f}%")

    except FileNotFoundError as e:
        print(f"\n  [SKIP] sub-{subject}: file not found -- {e}")
        continue

if len(ersp_list) == 0:
    print("No subjects loaded. Exiting.")
    raise SystemExit(1)

# Group average ERSP
ersp_group = np.mean(np.stack(ersp_list), axis=0)  # (n_freqs, 101)

# Group mean gait events
mean_lto_group = float(np.mean([e[0] for e in events_list]))
mean_lhs_group = float(np.mean([e[1] for e in events_list]))
mean_rto_group = float(np.mean([e[2] for e in events_list]))

n_subjects = len(ersp_list)

# Group topographies: pool subjects with matching channel count
n_ch_ref     = topo_list[0].shape[0]
valid_topo    = [t for t in topo_list    if t.shape[0] == n_ch_ref]
valid_weights = [w for w in weights_list if w.shape[0] == n_ch_ref]
group_topo    = np.mean(np.stack(valid_topo),    axis=0)   # (n_ch,)
group_weights = np.mean(np.stack(valid_weights), axis=0)   # (n_ch,)

print(f"\n  Group average: n={n_subjects} subjects")
print(f"  Group ERSP range: {ersp_group.min():.2f} / {ersp_group.max():.2f} dB")
print(f"  Group events: LTO={mean_lto_group:.1f}%  "
      f"LHS={mean_lhs_group:.1f}%  RTO={mean_rto_group:.1f}%")

# Build figure: left column [heatmap / event ticks / phase bars], right column [topo1 / topo2]
fig = plt.figure(figsize=(14, 7))
gs_outer = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.38, figure=fig)

gs_left = gridspec.GridSpecFromSubplotSpec(
    4, 1, subplot_spec=gs_outer[0],
    height_ratios=[10, 3, 1, 1], hspace=0,
)
gs_right = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_outer[1],
    hspace=0.50,
)

ax_heat   = fig.add_subplot(gs_left[0])
ax_wave   = fig.add_subplot(gs_left[1], sharex=ax_heat)
ax_events = fig.add_subplot(gs_left[2], sharex=ax_heat)
ax_phases = fig.add_subplot(gs_left[3], sharex=ax_heat)
ax_topo1  = fig.add_subplot(gs_right[0])
ax_topo2  = fig.add_subplot(gs_right[1])

fig.suptitle(
    f"Beta ERSP over Gait Cycle — Group Average  "
    f"(n={n_subjects}, Linear ROI, center=Cz)",
    fontsize=12, fontweight="bold", y=1.02,
)

# Row 0 — heatmap with event lines
im = ax_heat.imshow(
    ersp_group,
    aspect="auto",
    origin="lower",
    extent=[0, 100, FREQS[0] - 0.5, FREQS[-1] + 0.5],
    cmap="RdBu_r",
    vmin=-ERSP_CLIM, vmax=+ERSP_CLIM,
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

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.020, pad=0.01)
cbar.set_label("ERSP (dB)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax_heat.set_ylabel("Frequency (Hz)", fontsize=10)
ax_heat.set_xlim(0, 100)
ax_heat.set_ylim(FREQS[0] - 0.5, FREQS[-1] + 0.5)
ax_heat.set_yticks([13, 16, 20, 24, 28, 30])
ax_heat.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax_heat.tick_params(axis="y", labelsize=8)
ax_heat.spines["bottom"].set_visible(False)

# Row 1 — beta wavelet trace
beta_trace = ersp_group.mean(axis=0)   # (101,)
x_gait     = np.linspace(0, 100, 101)

ax_wave.plot(x_gait, beta_trace, color="#2c5f8a", linewidth=1.5)
ax_wave.fill_between(x_gait, beta_trace, 0,
                     where=(beta_trace >= 0), color="#c8392b", alpha=0.3)
ax_wave.fill_between(x_gait, beta_trace, 0,
                     where=(beta_trace < 0),  color="#2c5f8a", alpha=0.3)
ax_wave.axhline(0, color="grey", linewidth=0.8, linestyle="--")
for pct, _ in heatmap_events:
    ax_wave.axvline(pct, color="black", linewidth=1.0, linestyle=":", zorder=3)
ax_wave.set_ylabel("ERSP (dB)", fontsize=8)
ax_wave.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax_wave.spines["top"].set_visible(False)
ax_wave.spines["right"].set_visible(False)
ax_wave.tick_params(axis="y", labelsize=8)
y_abs      = float(np.abs(beta_trace).max())
tick_step  = max(1, int(np.ceil(y_abs / 2)))
ax_wave.set_yticks([-tick_step, 0, tick_step])

# Row 2 — event tick row
gait_event_ticks = [
    (0,               "RHS"),
    (mean_lto_group,  "LTO"),
    (mean_lhs_group,  "LHS"),
    (mean_rto_group,  "RTO"),
    (100,             "RHS"),
]

ax_events.set_ylim(-0.3, 1.0)
ax_events.plot([0, 100], [0.5, 0.5], color="black", linewidth=1.0, clip_on=False)

for pct, tick_label in gait_event_ticks:
    ax_events.plot([pct, pct], [0.2, 0.8], color="black", linewidth=1.0)
    ax_events.text(
        pct, 0.15, tick_label,
        ha="center", va="top",
        fontsize=8, color="black",
    )

ax_events.set_yticks([])
ax_events.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
for spine in ax_events.spines.values():
    spine.set_visible(False)

# Row 3 — phase bar row
DS_COLOR    = "#C4BFBF"   # dark grey — double support (IDS, FDS)
SWING_COLOR = "#D2D899"   # blue      — swing (RLS, LLS)

phase_bars = [
    (0,               mean_lto_group,                    DS_COLOR,    "IDS"),
    (mean_lto_group,  mean_lhs_group - mean_lto_group,   SWING_COLOR, "LLS"),
    (mean_lhs_group,  mean_rto_group - mean_lhs_group,   DS_COLOR,    "FDS"),
    (mean_rto_group,  100 - mean_rto_group,               SWING_COLOR, "RLS"),
]

ax_phases.set_ylim(0, 1)
for left, width, color, label in phase_bars:
    ax_phases.barh(0.5, width, left=left, height=1.0, color=color,
                   align="center", edgecolor="none")
    ax_phases.text(
        left + width / 2, 0.5, label,
        ha="center", va="center",
        fontsize=8, color="black",
    )

ax_phases.set_xlabel("Gait cycle (%)", fontsize=10, labelpad=4)
ax_phases.set_yticks([])
ax_phases.set_xticks([0, 20, 40, 60, 80, 100])
ax_phases.tick_params(axis="x", labelsize=8)
for spine in ax_phases.spines.values():
    spine.set_visible(False)

# Panel 2 — Beta power topography
vlim_topo = np.abs(group_topo).max()
im2, *_ = mne.viz.plot_topomap(
    group_topo, info_ref,
    axes=ax_topo1,
    show=False,
    cmap="RdBu_r",
    vlim=(-vlim_topo, vlim_topo),
    contours=4,
)
fig.colorbar(im2, ax=ax_topo1, fraction=0.046, pad=0.04).set_label("dB", fontsize=8)
ax_topo1.set_title("Beta power\ntopography", fontsize=10)

# Panel 3 — Spatial filter topography
im3, *_ = mne.viz.plot_topomap(
    group_weights, info_ref,
    axes=ax_topo2,
    show=False,
    cmap="Reds",
    vlim=(0, group_weights.max()),
    contours=4,
)
fig.colorbar(im3, ax=ax_topo2, fraction=0.046, pad=0.04).set_label("Weight", fontsize=8)
ax_topo2.set_title("Spatial filter\n(Linear ROI weights)", fontsize=10)

out_path = PLOTS_DIR / "group_beta_ersp_topo.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Group figure saved -> {out_path.name}")
