"""
ana06_plotbetagait.py
=====================
Plot the group-average extended-band ERSP (8-60 Hz -- see
src.config.PIPELINE_TFR_FMAX) across the normalised gait cycle, alongside
three beta-band (13-30 Hz) topographies.

Left: an 8-60 Hz ERSP heatmap (beta and the 40-60 Hz range both visible;
the 50 Hz notch is not annotated on-figure -- see band_diagnostics.txt
and the manuscript Discussion) with event ticks and gait phase bars
underneath.

Right: three beta-band (13-30 Hz) topographies on a shared colour scale
with one shared colorbar: whole gait cycle, swing only, double-stance
only.

Input
-----
d05_ersp/       sub-{sub}_ersp_beta.npy        (n_ch x n_freqs x 101)
                sub-{sub}_roi_weights.npy       (n_ch,)
d04_gaitepochs/ sub-{sub}_cycles_kept.tsv      (for mean gait event positions)
d03_clean/      sub-{sub}_desc-icaClean_concat_raw.fif  (for channel info)

Output
------
results/pipeline/<dataset>/plots/<dataset>_betaersp_gait.png
Standing-baseline is the only normalization for both datasets
(2026-08-11 -- GPM removed project-wide; it was exploratory leftover
from an earlier "compute both to compare" phase, mathematically
identical to standing for the double-stance-vs-swing contrast, see
NOTES.md).
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
from src.config import (
    DATASET, SUBJECTS, DIR_PLOTS, PIPELINE_TFR_FMAX, ERSP_CMAP,
    TFR_FMIN, ROI_CENTER_CH,
)
from src.spatial_filter import linear_roi_weights, apply_linear_roi, TOPOMAP_SPHERE
from src.ersp import (
    load_reference_anchors, phase_split_indices, BETA_FMIN, BETA_FMAX,
)

NORM_LABEL = "dB relative to standing baseline"

FREQS = np.arange(TFR_FMIN, int(PIPELINE_TFR_FMAX) + 1)

dirs          = get_dataset_dirs(DATASET)
ERSP_DIR      = dirs["ersp"]
CLEAN_DIR     = dirs["clean"]
GAITEPOCH_DIR = dirs["gaitepochs"]
PLOTS_DIR     = Path(DIR_PLOTS)

# Group-median gait-event anchors, written once by prepana05 (same values
# used there to warp every cycle onto the common grid). Loaded rather than
# recomputed here so the plotted event lines always match the anchors the
# ERSP arrays were actually warped to.
A_lto, A_lhs, A_rto = load_reference_anchors(ERSP_DIR)

# Event-anchored double-stance / swing phase windows on the common 101-point
# grid -- same anchors and construction as prepana05/multiverse_pipeline
# (src.ersp.phase_split_indices).
double_stance_idx, swing_idx = phase_split_indices((A_lto, A_lhs, A_rto), n_points=101)
BETA_MASK = (FREQS >= BETA_FMIN) & (FREQS <= BETA_FMAX)

ersp_list       = []
events_list     = []
beta_whole_list = []   # per-subject (n_ch,) beta-band ERSP, whole gait cycle
beta_swing_list = []   # per-subject (n_ch,) beta-band ERSP, swing only
beta_ds_list    = []   # per-subject (n_ch,) beta-band ERSP, double-stance only
info_ref        = None

for subject in SUBJECTS:
    try:
        ersp_path   = ERSP_DIR      / f"sub-{subject}_ersp_beta.npy"
        cycles_path = GAITEPOCH_DIR / f"sub-{subject}_cycles_kept.tsv"
        clean_path  = CLEAN_DIR     / f"sub-{subject}_desc-icaClean_concat_raw.fif"

        ersp     = np.load(ersp_path)   # (n_ch, n_freqs, 101), standing-baselined
        cycles   = pd.read_csv(cycles_path, sep="\t")
        raw_ref  = mne.io.read_raw_fif(clean_path, preload=False, verbose=False)

        # Load or compute linear ROI weights
        weights_path = ERSP_DIR / f"sub-{subject}_roi_weights.npy"
        if weights_path.exists():
            sub_weights = np.load(weights_path)
        else:
            sub_weights = linear_roi_weights(raw_ref.info, center_ch=ROI_CENTER_CH)

        # Apply weights: (n_ch, n_freqs, 101) → (n_freqs, 101)
        ersp_roi = apply_linear_roi(ersp, sub_weights)

        # Per-channel beta-band (13-30 Hz) topographies, whole cycle / swing /
        # double-stance, using the same phase windows as prepana05/src.ersp.
        # phase_split_indices -- no new windowing invented.
        ersp_beta = ersp[:, BETA_MASK, :]                        # (n_ch, n_beta, 101)
        beta_whole_list.append(ersp_beta.mean(axis=(1, 2)))                    # (n_ch,)
        beta_swing_list.append(ersp_beta[:, :, swing_idx].mean(axis=(1, 2)))   # (n_ch,)
        beta_ds_list.append(ersp_beta[:, :, double_stance_idx].mean(axis=(1, 2)))  # (n_ch,)

        dur      = cycles["rhs_end_s"].values - cycles["rhs_start_s"].values
        lto_pct  = (cycles["lto_s"].values  - cycles["rhs_start_s"].values) / dur * 100
        lhs_pct  = (cycles["lhs_s"].values  - cycles["rhs_start_s"].values) / dur * 100
        rto_pct  = (cycles["rto_s"].values  - cycles["rhs_start_s"].values) / dur * 100
        mean_lto = float(np.mean(lto_pct))
        mean_lhs = float(np.mean(lhs_pct))
        mean_rto = float(np.mean(rto_pct))

        ersp_list.append(ersp_roi)
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

# Group gait-event positions -- the shared group-median anchors from
# prepana05 (every cycle was warped to land exactly on these percentages).
mean_lto_group, mean_lhs_group, mean_rto_group = A_lto, A_lhs, A_rto

n_subjects = len(ersp_list)

# Group beta topographies: pool subjects with matching channel count
n_ch_ref         = beta_whole_list[0].shape[0]
valid_beta_whole = [t for t in beta_whole_list if t.shape[0] == n_ch_ref]
valid_beta_swing = [t for t in beta_swing_list if t.shape[0] == n_ch_ref]
valid_beta_ds    = [t for t in beta_ds_list    if t.shape[0] == n_ch_ref]
group_beta_whole = np.mean(np.stack(valid_beta_whole), axis=0)   # (n_ch,)
group_beta_swing = np.mean(np.stack(valid_beta_swing), axis=0)   # (n_ch,)
group_beta_ds    = np.mean(np.stack(valid_beta_ds),    axis=0)   # (n_ch,)

print(f"\n  Group average: n={n_subjects} subjects")
print(f"  Group ERSP range: {ersp_group.min():.2f} / {ersp_group.max():.2f} dB")
print(f"  Group events: LTO={mean_lto_group:.1f}%  "
      f"LHS={mean_lhs_group:.1f}%  RTO={mean_rto_group:.1f}%")

out_path = PLOTS_DIR / f"{DATASET}_betaersp_gait.png"

fig = plt.figure(figsize=(18, 13))
gs_outer = gridspec.GridSpec(1, 2, width_ratios=[2.6, 1.0], wspace=0.3, figure=fig)

gs_left = gridspec.GridSpecFromSubplotSpec(
    4, 1, subplot_spec=gs_outer[0],
    height_ratios=[10, 3, 1, 1], hspace=0.15,
)
gs_right = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_outer[1],
    hspace=0.45,
)

ax_heat    = fig.add_subplot(gs_left[0])
ax_beta    = fig.add_subplot(gs_left[1], sharex=ax_heat)
ax_events  = fig.add_subplot(gs_left[2], sharex=ax_heat)
ax_phases  = fig.add_subplot(gs_left[3], sharex=ax_heat)
ax_topo_w  = fig.add_subplot(gs_right[0])
ax_topo_sw = fig.add_subplot(gs_right[1])
ax_topo_ds = fig.add_subplot(gs_right[2])

fig.suptitle(
    f"Extended-band ERSP over Gait Cycle — Group Average  "
    f"(n={n_subjects}, Linear ROI, center=Cz)\n"
    f"Normalization: {NORM_LABEL}",
    fontsize=17, fontweight="bold", y=1.03,
)

# Heatmap — 8-PIPELINE_TFR_FMAX Hz ERSP
# x-extent half-bin correction: ersp_group has N_POINTS=101 columns at
# TRUE gait-cycle positions x=0,1,...,100 (see src.ersp.warp_cycle_to_grid's
# np.linspace(0,100,101)) -- imshow treats an n-column array passed with
# extent=[0,100] as n equal-width BINS spanning that range, so column i's
# rendered center lands at (i+0.5)*100/n, not at its true value i. That
# mismatched the event lines/phase bar below (drawn at the true anchor
# percentages), by up to +-0.495%. Half-bin correcting the extent (as the
# y/frequency axis already correctly does with +-0.5 Hz) makes column i
# render centered exactly on x=i, matching the lines/phase bar exactly.
n_gait_points = ersp_group.shape[-1]
dx_gait       = 100.0 / (n_gait_points - 1)
_uncorrected_max_offset = dx_gait / 2   # what column 0/-1 would be off by without this fix
print(f"\n  Alignment check -- event anchors: LTO={mean_lto_group:.2f}%  "
      f"LHS={mean_lhs_group:.2f}%  RTO={mean_rto_group:.2f}%  RHS=0.00/100.00%")
print(f"  Heatmap x-extent: uncorrected=[0, 100] (max column-vs-anchor "
      f"offset +-{_uncorrected_max_offset:.4f}%) -> corrected="
      f"[{-dx_gait/2:.4f}, {100 + dx_gait/2:.4f}] (offset now 0.0000% at every column)")
# Symmetric color limit, computed PER DATASET from this run's own data
# (99th percentile of |value|) -- NOT a single global constant shared
# across datasets. Jacobsen's real range (~+-0.8 dB) is roughly 4x
# smaller than stepUpAms's (~+-3 dB); a shared fixed limit tuned for one
# made the other render nearly blank.
heatmap_vlim = float(np.percentile(np.abs(ersp_group), 99))
print(f"  Heatmap color limit ({DATASET}): "
      f"99th pct |value| = +-{heatmap_vlim:.3f} dB")

im = ax_heat.imshow(
    ersp_group,
    aspect="auto",
    origin="lower",
    extent=[-dx_gait / 2, 100 + dx_gait / 2, FREQS[0] - 0.5, FREQS[-1] + 0.5],
    cmap=ERSP_CMAP,
    vmin=-heatmap_vlim, vmax=+heatmap_vlim,
    zorder=1,
)

heatmap_events = [
    (0,               "RHS"),
    (mean_lto_group,  "LTO"),
    (mean_lhs_group,  "LHS"),
    (mean_rto_group,  "RTO"),
    (100,             "RHS"),
]
for pct, _ in heatmap_events:
    ax_heat.axvline(pct, color="black", linewidth=1.0, linestyle=":", zorder=3)

# Beta band boundaries (13-30 Hz)
ax_heat.axhline(BETA_FMIN, color="white", linewidth=0.8, linestyle="--", alpha=0.6, zorder=4)
ax_heat.axhline(BETA_FMAX, color="white", linewidth=0.8, linestyle="--", alpha=0.6, zorder=4)

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.020, pad=0.01)
cbar.set_label("ERSP (dB)", fontsize=13)
cbar.ax.tick_params(labelsize=11)

ax_heat.set_ylabel("Frequency (Hz)", fontsize=14)
ax_heat.set_xlim(0, 100)
ax_heat.set_ylim(FREQS[0] - 0.5, FREQS[-1] + 0.5)
ax_heat.set_yticks(sorted(set([8, 13, 20, 30, 40, 50, int(PIPELINE_TFR_FMAX)])))
ax_heat.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax_heat.tick_params(axis="y", labelsize=11)
ax_heat.spines["bottom"].set_visible(False)

# Beta-band (13-30 Hz) trace: collapse the beta rows of the same group
# ERSP to one line across the gait cycle, x-aligned with the heatmap.
beta_trace = ersp_group[BETA_MASK, :].mean(axis=0)   # (101,)
x_gait     = np.linspace(0, 100, n_gait_points)

ax_beta.plot(x_gait, beta_trace, color="#2c5f8a", linewidth=1.8, zorder=2)
ax_beta.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
for pct, _ in heatmap_events:
    ax_beta.axvline(pct, color="black", linewidth=1.0, linestyle=":", zorder=3)
ax_beta.set_ylabel("Beta (13-30 Hz)\nERSP (dB)", fontsize=11)
ax_beta.set_xlim(0, 100)
ax_beta.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax_beta.tick_params(axis="y", labelsize=10)
ax_beta.spines["top"].set_visible(False)
ax_beta.spines["right"].set_visible(False)

# Event tick row
ax_events.set_ylim(-0.3, 1.0)
ax_events.plot([0, 100], [0.5, 0.5], color="black", linewidth=1.0, clip_on=False)
for pct, tick_label in heatmap_events:
    ax_events.plot([pct, pct], [0.2, 0.8], color="black", linewidth=1.0)
    ax_events.text(pct, 0.15, tick_label, ha="center", va="top", fontsize=11, color="black")
ax_events.set_yticks([])
ax_events.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
for spine in ax_events.spines.values():
    spine.set_visible(False)

# Phase bar row
DS_COLOR    = "#C4BFBF"
SWING_COLOR = "#D2D899"
phase_bars = [
    (0,               mean_lto_group,                   DS_COLOR,    "DS"),
    (mean_lto_group,  mean_lhs_group - mean_lto_group,  SWING_COLOR, "LLS"),
    (mean_lhs_group,  mean_rto_group - mean_lhs_group,  DS_COLOR,    "DS"),
    (mean_rto_group,  100 - mean_rto_group,              SWING_COLOR, "RLS"),
]
print(f"  Phase-bar boundaries: DS 0.00->{mean_lto_group:.2f}  "
      f"LLS {mean_lto_group:.2f}->{mean_lhs_group:.2f}  "
      f"DS {mean_lhs_group:.2f}->{mean_rto_group:.2f}  "
      f"RLS {mean_rto_group:.2f}->100.00  (derived from the same A_lto/A_lhs/A_rto "
      f"anchors as the vertical lines above -- cannot drift apart)")
ax_phases.set_ylim(0, 1)
for left, width, color, label in phase_bars:
    ax_phases.barh(0.5, width, left=left, height=1.0, color=color, align="center", edgecolor="none")
    ax_phases.text(left + width / 2, 0.5, label, ha="center", va="center", fontsize=11, color="black")
ax_phases.set_xlabel("Gait cycle (%)", fontsize=14, labelpad=4)
ax_phases.set_yticks([])
ax_phases.set_xticks([0, 20, 40, 60, 80, 100])
ax_phases.tick_params(axis="x", labelsize=11)
for spine in ax_phases.spines.values():
    spine.set_visible(False)

# Three beta-band topographies, shared colour scale + one shared colorbar:
# whole gait cycle, swing only, double-stance only.
topo_arrays_shown = [group_beta_whole, group_beta_swing, group_beta_ds]
topo_specs = [
    (ax_topo_w,  group_beta_whole, "Beta (13-30 Hz)\nwhole gait cycle"),
    (ax_topo_sw, group_beta_swing, "Beta (13-30 Hz)\nswing only"),
    (ax_topo_ds, group_beta_ds,    "Beta (13-30 Hz)\ndouble-stance only"),
]

# Symmetric color limit, computed PER DATASET from the 99th percentile of
# |value| across exactly the arrays actually shown in this figure (not a
# global constant -- see heatmap_vlim above).
vlim_beta = float(np.percentile(np.abs(np.concatenate(topo_arrays_shown)), 99))
print(f"  Beta topo color limit ({DATASET}): "
      f"99th pct |value| = +-{vlim_beta:.3f} dB")

im_topo = None
for ax, data, title in topo_specs:
    im_topo, *_ = mne.viz.plot_topomap(
        data, info_ref,
        axes=ax,
        show=False,
        cmap=ERSP_CMAP,
        vlim=(-vlim_beta, vlim_beta),
        contours=4,
        sphere=TOPOMAP_SPHERE,
    )
    ax.set_title(title, fontsize=14)

topo_cbar = fig.colorbar(
    im_topo, ax=[ax_topo_w, ax_topo_sw, ax_topo_ds],
    fraction=0.05, pad=0.08,
)
topo_cbar.set_label("Beta ERSP (dB)", fontsize=12)
topo_cbar.ax.tick_params(labelsize=10)

fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n  Group figure saved -> {out_path.name}")
