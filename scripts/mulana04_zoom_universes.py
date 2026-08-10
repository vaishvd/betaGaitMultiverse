"""
mulana04_zoom_universes.py
===========================
"Zoom-in" figure: the beta-band-inclusive ERSP heatmap over the gait
cycle for the LOWEST, MEDIAN, and HIGHEST double-stance-minus-swing beta
effect-size universes out of the completed stepUpAms 27-universe
multiverse (highpass_hz x asr_mode x iclabel_rule -- see
mulana01_create_multiverse.py). Visualizes how the effect magnitude
shrinks across specifications while the stance-vs-swing pattern
(hopefully) persists.

READ-ONLY on results/multiverse/. This script:
  - reads per-universe group results from
    results/multiverse/stepup/comet/scripts/temp/universe_*.pkl
    (written once by mulana02_run_multiverse.py -- never re-run here)
  - reads each selected universe's already-cached, ICA-cleaned raw
    (results/multiverse/stepup/branches/<subj>/<branch>/sub-*_desc-
    icaClean_<rule>_raw.fif) directly via mne.io.read_raw_fif
  - never calls ICA fit/apply and never writes anything under
    results/multiverse/ -- if a required cached file is missing for one
    of the three selected universes, this script raises rather than
    computing/caching it (that would mean re-running a universe, which
    is out of scope here).
The remaining per-subject steps (standing-baseline power, gait-cycle
warp, ROI reduction) are pure in-memory recomputation from that cached
input, reusing the same shared helpers as the canonical pipeline
(src/ersp.py) and the multiverse pipeline (src/multiverse_pipeline.py)
so results stay numerically consistent -- but this script duplicates
multiverse_pipeline.run_subject_multiverse()'s body rather than importing
it, because that function returns only the beta-band scalar contrast
(what's cached per universe); it never computes or saves the full
frequency x gait-cycle ROI array needed for a heatmap, so there's
nothing to "not re-run" for that part -- it's new computation needed
only for this figure, done once per subject for the 3 selected universes
(not one of the 27 decision-defining runs).

Output
------
results/pipeline/stepup/plots/stepup_multiverse_zoom.png
"""

import pickle

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import ttest_1samp

from src.config import (
    DATASET, DIR_PLOTS, DIR_MULTIVERSE_COMET, MULTIVERSE_LOWPASS_HZ,
    ERSP_CMAP, ERSP_HEATMAP_VLIM, ROI_CENTER_CH,
)
from src.config import MULTIVERSE_SUBJECTS as SUBJECTS
from src.paths import get_dataset_dirs
from src.multiverse_pipeline import (
    _branch_dir, FREQS, N_CYCLES_WAV, N_POINTS, EDGE_CROP, AMP_THRESH,
)
from src.ersp import (
    load_group_anchors, warp_cycle_to_grid, phase_split_indices,
    compute_standing_baseline, BETA_FMIN, BETA_FMAX,
)
from src.spatial_filter import linear_roi_weights, apply_linear_roi

dirs      = get_dataset_dirs(DATASET)
EVENT_DIR = dirs["gait_events"]
ERSP_DIR  = dirs["ersp"]
PLOTS_DIR = Path(DIR_PLOTS)

# comet.Multiverse.run() combines each universe_N.pkl (written under
# comet/scripts/temp/ while running) into this single file after a
# successful run, deleting the individual per-universe files -- so this
# is what's actually on disk once mulana02_run_multiverse.py has
# completed, not the temp/ directory.
COMBINED_RESULTS_PATH = DIR_MULTIVERSE_COMET / "results" / "multiverse_results.pkl"


def _decisions_dict(raw_decisions):
    """{'Decision 1': 'highpass_hz', 'Value 1': 0.5, ...} -> {'highpass_hz': 0.5, ...}"""
    out, i = {}, 1
    while f"Decision {i}" in raw_decisions:
        out[raw_decisions[f"Decision {i}"]] = raw_decisions[f"Value {i}"]
        i += 1
    return out


def subject_universe_roi_ersp(subject: str, decisions: dict) -> np.ndarray:
    """
    Full ROI-weighted ERSP array (n_freqs, N_POINTS) for one subject under
    one universe's decisions, over the group anchor-warped gait cycle.

    READ-ONLY: loads this universe's already-cached ICA-cleaned raw
    directly (mne.io.read_raw_fif) -- never fits/applies ICA, never
    writes. Raises FileNotFoundError if the cached file doesn't exist
    (mulana02_run_multiverse.py must have already produced it).
    """
    branch_dir = _branch_dir(subject, decisions)   # read-only: same key, dir already exists
    iclabel_rule = decisions["iclabel_rule"]
    iclean_path = branch_dir / f"sub-{subject}_desc-icaClean_{iclabel_rule}_raw.fif"
    if not iclean_path.exists():
        raise FileNotFoundError(
            f"sub-{subject}: no cached cleaned raw at {iclean_path} -- "
            "run mulana02_run_multiverse.py for this universe first."
        )
    raw_clean = mne.io.read_raw_fif(iclean_path, preload=True, verbose=False)

    def crop(r, desc):
        ann = [a for a in r.annotations if a["description"] == desc][0]
        return r.copy().crop(ann["onset"], min(ann["onset"] + ann["duration"], r.times[-1]))

    raw_stand = crop(raw_clean, "STAND")
    stand_tmax = raw_stand.times[-1] - 2.0
    if stand_tmax <= 0:
        raise RuntimeError(f"sub-{subject}: standing segment too short after trimming")
    raw_stand = raw_stand.crop(tmax=stand_tmax)

    raw_walk = crop(raw_clean, "CS")
    sfreq    = raw_walk.info["sfreq"]

    baseline_power = compute_standing_baseline(
        raw_stand, FREQS, N_CYCLES_WAV, edge_crop=EDGE_CROP, amp_thresh=AMP_THRESH,
    )

    A_lto, A_lhs, A_rto = load_group_anchors(ERSP_DIR)
    cycles    = pd.read_csv(EVENT_DIR / f"sub-{subject}_cycles.tsv", sep="\t")
    walk_data = raw_walk.get_data()

    tfr_cycles = []
    for _, row in cycles.iterrows():
        i0 = int(round(row["rhs_start_s"] * sfreq))
        i1 = int(round(row["rhs_end_s"]   * sfreq))
        if i0 < 0 or i1 > walk_data.shape[1] or i1 <= i0:
            continue
        seg = walk_data[:, i0:i1]
        if not np.isfinite(seg).all() or np.max(np.abs(seg)) > AMP_THRESH:
            continue

        power = mne.time_frequency.tfr_array_morlet(
            seg[np.newaxis], sfreq=sfreq, freqs=FREQS,
            n_cycles=N_CYCLES_WAV, output="power",
            zero_mean=True, verbose=False,
        )[0]
        crop_n = int(EDGE_CROP * power.shape[-1])
        power = power[..., crop_n:-crop_n] if crop_n else power

        lto_idx = int(round(row["lto_s"] * sfreq)) - i0 - crop_n
        lhs_idx = int(round(row["lhs_s"] * sfreq)) - i0 - crop_n
        rto_idx = int(round(row["rto_s"] * sfreq)) - i0 - crop_n

        try:
            warped = warp_cycle_to_grid(
                power, lto_idx, lhs_idx, rto_idx,
                anchors=(A_lto, A_lhs, A_rto), n_points=N_POINTS,
            )
        except ValueError:
            continue
        tfr_cycles.append(warped)

    if len(tfr_cycles) < 20:
        raise RuntimeError(f"sub-{subject}: only {len(tfr_cycles)} gait cycles accepted")

    tfr_stack = np.stack(tfr_cycles)   # (n_cycles, n_ch, n_freqs, N_POINTS)
    ersp_per_cycle = 10 * np.log10(
        tfr_stack / baseline_power[np.newaxis, :, :, np.newaxis]
    )
    ersp_mean = ersp_per_cycle.mean(axis=0)   # (n_ch, n_freqs, N_POINTS)

    weights = linear_roi_weights(raw_clean.info, center_ch=ROI_CENTER_CH)
    return apply_linear_roi(ersp_mean, weights)   # (n_freqs, N_POINTS)


# ---------------------------------------------------------------------
# 1. Load per-universe cached group results (read-only)
# ---------------------------------------------------------------------
if not COMBINED_RESULTS_PATH.exists():
    raise SystemExit(
        f"No combined multiverse results found at {COMBINED_RESULTS_PATH} -- "
        "run mulana02_run_multiverse.py first."
    )
with open(COMBINED_RESULTS_PATH, "rb") as f:
    all_universes = pickle.load(f)

universe_keys = sorted(all_universes.keys(), key=lambda k: int(k.split("_")[1]))

records = []
for key in universe_keys:
    data = all_universes[key]
    universe_num = int(key.split("_")[1])

    beta_diffs = np.asarray(data["t_stats"], dtype=float)  # per-subject dB diffs
    beta_diffs = beta_diffs[np.isfinite(beta_diffs)]
    if beta_diffs.size < 2:
        print(f"  [SKIP] universe_{universe_num}: only {beta_diffs.size} usable subjects")
        continue

    effect_db = float(beta_diffs.mean())
    t_stat, p_val = ttest_1samp(beta_diffs, 0.0)
    saved_t = data.get("group_t_mean", float("nan"))
    if np.isfinite(saved_t) and not np.isclose(t_stat, saved_t, atol=1e-6):
        print(f"  [WARN] universe_{universe_num}: recomputed t={t_stat:.4f} "
              f"!= cached group_t_mean={saved_t:.4f}")

    decisions = _decisions_dict(data["__decisions"])
    records.append({
        "universe":    universe_num,
        "highpass_hz": decisions["highpass_hz"],
        "asr_mode":    decisions["asr_mode"],
        "iclabel_rule": decisions["iclabel_rule"],
        "effect_db":   effect_db,
        "t_stat":      float(t_stat),
        "p_val":       float(p_val),
        "n_subjects":  int(beta_diffs.size),
    })

if len(records) < 3:
    raise SystemExit(f"Only {len(records)} usable universes found -- need at least 3.")

df = pd.DataFrame(records).sort_values("effect_db").reset_index(drop=True)
n = len(df)
median_idx = (n - 1) // 2   # for odd n (27), this is exactly the middle value -- no tie-break needed

print(f"\n  {n} universes with usable results (of {len(all_universes)} total).")
print(f"  Median rule: n={n} is odd, so the median is the single middle "
      f"value at sorted position {median_idx + 1} of {n} (0-indexed {median_idx}); "
      f"no tie-breaking needed.\n")

selection = {
    "LOWEST":  df.iloc[0],
    "MEDIAN":  df.iloc[median_idx],
    "HIGHEST": df.iloc[-1],
}

print("  Selected universes:")
for label, row in selection.items():
    print(f"    {label:8s} universe_{int(row['universe'])}: "
          f"HP={row['highpass_hz']} Hz, ASR={row['asr_mode']}, "
          f"IC={row['iclabel_rule']}  |  "
          f"{row['effect_db']:+.3f} dB  t={row['t_stat']:.3f}  "
          f"p={row['p_val']:.4g}  (n={int(row['n_subjects'])} subjects)")

# ---------------------------------------------------------------------
# 2. Group-average ROI ERSP heatmap for each of the 3 selected universes
#    (new computation -- not part of the 27 cached decision-defining
#    runs, needed only because no full ERSP array is cached per universe)
# ---------------------------------------------------------------------
A_lto, A_lhs, A_rto = load_group_anchors(ERSP_DIR)
double_stance_idx, swing_idx = phase_split_indices((A_lto, A_lhs, A_rto), n_points=N_POINTS)

group_ersp = {}
for label, row in selection.items():
    decisions = {
        "highpass_hz":  float(row["highpass_hz"]),
        "asr_mode":     row["asr_mode"],
        "iclabel_rule": row["iclabel_rule"],
        "lowpass_hz":   MULTIVERSE_LOWPASS_HZ,
    }
    print(f"\n  Computing group ERSP for {label} (universe_{int(row['universe'])}) ...")
    per_subj = []
    for subject in SUBJECTS:
        try:
            per_subj.append(subject_universe_roi_ersp(subject, decisions))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"    [SKIP] sub-{subject}: {e}")
    if len(per_subj) < 2:
        raise SystemExit(f"{label}: only {len(per_subj)} usable subjects -- cannot average.")
    group_ersp[label] = np.mean(np.stack(per_subj), axis=0)   # (n_freqs, N_POINTS)
    print(f"    -> n={len(per_subj)} subjects averaged")

# ---------------------------------------------------------------------
# 3. Shared symmetric colour scale across all three panels
# ---------------------------------------------------------------------
# ERSP_HEATMAP_VLIM (src.config) is the SAME fixed constant the
# reference-pipeline ERSP heatmap uses -- using one shared constant
# (rather than each figure/run computing its own data-driven max)
# guarantees these 3 panels share one scale by construction, and keeps
# this figure's dB scale directly comparable to the reference figure's.
vlim_datadriven = max(np.abs(arr).max() for arr in group_ersp.values())
vmin, vmax = -ERSP_HEATMAP_VLIM, ERSP_HEATMAP_VLIM
print(f"\n  Shared color scale: previous data-driven=+-{vlim_datadriven:.3f} dB  "
      f"-> fixed shared vmin={vmin:.3f} dB, vmax={vmax:.3f} dB "
      f"(same ERSP_HEATMAP_VLIM as the reference-pipeline figure)")

# ---------------------------------------------------------------------
# 4. Build the 3-panel figure
# ---------------------------------------------------------------------
out_path = PLOTS_DIR / f"{DATASET}_multiverse_zoom.png"

order = ["LOWEST", "MEDIAN", "HIGHEST"]
fig = plt.figure(figsize=(20, 7))
gs_outer = gridspec.GridSpec(1, len(order), wspace=0.35, figure=fig)

heatmap_events = [
    (0,      "RHS"),
    (A_lto,  "LTO"),
    (A_lhs,  "LHS"),
    (A_rto,  "RTO"),
    (100,    "RHS"),
]
DS_COLOR    = "#C4BFBF"
SWING_COLOR = "#D2D899"
phase_bars = [
    (0,      A_lto,          DS_COLOR,    "DS"),
    (A_lto,  A_lhs - A_lto,  SWING_COLOR, "LLS"),
    (A_lhs,  A_rto - A_lhs,  DS_COLOR,    "DS"),
    (A_rto,  100 - A_rto,    SWING_COLOR, "RLS"),
]

im = None
for col, label in enumerate(order):
    row = selection[label]
    ersp = group_ersp[label]

    gs_col = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_outer[col],
        height_ratios=[10, 1, 1], hspace=0.12,
    )
    ax_heat   = fig.add_subplot(gs_col[0])
    ax_events = fig.add_subplot(gs_col[1], sharex=ax_heat)
    ax_phases = fig.add_subplot(gs_col[2], sharex=ax_heat)

    # x-extent half-bin correction -- same fix as the reference-pipeline
    # figure (see prepana06_plotbetagait.py): an n-column array plotted
    # with extent=[0,100] renders column i centered at (i+0.5)*100/n, not
    # at its true value i, mismatching the event lines/phase bar below by
    # up to +-0.5%. Correcting the extent removes that offset entirely.
    n_gait_points = ersp.shape[-1]
    dx_gait       = 100.0 / (n_gait_points - 1)
    im = ax_heat.imshow(
        ersp, aspect="auto", origin="lower",
        extent=[-dx_gait / 2, 100 + dx_gait / 2, FREQS[0] - 0.5, FREQS[-1] + 0.5],
        cmap=ERSP_CMAP, vmin=vmin, vmax=vmax, zorder=1,
    )
    for pct, _ in heatmap_events:
        ax_heat.axvline(pct, color="black", linewidth=1.0, linestyle=":", zorder=3)
    ax_heat.axhline(BETA_FMIN, color="white", linewidth=0.8, linestyle="--", alpha=0.6, zorder=4)
    ax_heat.axhline(BETA_FMAX, color="white", linewidth=0.8, linestyle="--", alpha=0.6, zorder=4)

    ax_heat.set_xlim(0, 100)
    ax_heat.set_ylim(FREQS[0] - 0.5, FREQS[-1] + 0.5)
    ax_heat.set_yticks(sorted(set([8, 13, 20, 30, int(FREQS[-1])])))
    ax_heat.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_heat.tick_params(axis="y", labelsize=9)
    ax_heat.spines["bottom"].set_visible(False)
    if col == 0:
        ax_heat.set_ylabel("Frequency (Hz)", fontsize=11)

    ax_heat.set_title(
        f"{label}: HP={row['highpass_hz']} Hz, ASR={row['asr_mode']}, "
        f"IC={row['iclabel_rule']}\n{row['effect_db']:+.2f} dB  "
        f"(t={row['t_stat']:.2f}, p={row['p_val']:.3g})",
        fontsize=11,
    )

    ax_events.set_ylim(-0.3, 1.0)
    ax_events.plot([0, 100], [0.5, 0.5], color="black", linewidth=1.0, clip_on=False)
    for pct, tick_label in heatmap_events:
        ax_events.plot([pct, pct], [0.2, 0.8], color="black", linewidth=1.0)
        ax_events.text(pct, 0.15, tick_label, ha="center", va="top", fontsize=8, color="black")
    ax_events.set_yticks([])
    ax_events.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine in ax_events.spines.values():
        spine.set_visible(False)

    ax_phases.set_ylim(0, 1)
    for left, width, color, plabel in phase_bars:
        ax_phases.barh(0.5, width, left=left, height=1.0, color=color, align="center", edgecolor="none")
        ax_phases.text(left + width / 2, 0.5, plabel, ha="center", va="center", fontsize=8, color="black")
    ax_phases.set_xlabel("Gait cycle (%)", fontsize=10, labelpad=4)
    ax_phases.set_yticks([])
    ax_phases.set_xticks([0, 20, 40, 60, 80, 100])
    ax_phases.tick_params(axis="x", labelsize=8)
    for spine in ax_phases.spines.values():
        spine.set_visible(False)

cbar = fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.02)
cbar.set_label("ERSP (dB)", fontsize=10)

fig.suptitle(
    "Beta ERSP across lowest / median / highest multiverse specifications",
    fontsize=13, fontweight="bold", y=1.02,
)

fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\n  Zoom-in figure saved -> {out_path}")
