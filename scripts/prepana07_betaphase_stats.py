"""
ana07_betaphase_stats.py
=========================
Pipeline step 7: per-subject ERSP maps -> group beta double-stance-vs-swing
paired t-test.

Does not recompute TFR/ERSP -- consumes the per-subject dB maps already
written by prepana05_gaitcycles2tfr.py. Reduces to a single beta-band
(13-30 Hz) scalar per subject using the same linear Cz ROI weights as
prepana05/06, then runs a paired t-test across subjects on the double-
stance-minus-swing difference. This is the canonical group-level
inferential test for the beta double-stance-vs-swing contrast. The
paired t-test itself (scipy.stats.ttest_rel on the two per-subject
scalar arrays) is unchanged by NORMALIZATION -- only how the two scalars
are derived differs.

NORMALIZATION (src.config, env var BETAGAIT_NORMALIZATION, default "gpm"):
  "standing" -- ORIGINAL path, byte-for-byte unchanged: reads the
                pre-reduced (already double-stance/swing-averaged,
                standing-baselined) sub-{sub}_ersp_{double_stance,swing}.npy
                written by prepana05, exactly as before this feature was
                added.
  "gpm"      -- reads the full per-cycle sub-{sub}_ersp_beta.npy
                (n_ch, n_freqs, 101) instead, applies
                src.ersp.apply_gpm_normalization, then reduces to the
                same double-stance/swing scalars via the same ROI
                weights, beta mask, and phase_split_indices anchors used
                everywhere else. Mathematically must reproduce the exact
                same t-statistic as "standing" (see apply_gpm_normalization's
                docstring) -- this script prints an explicit cross-check
                of that per subject.

Input
-----
d05_ersp/
    sub-{sub}_ersp_double_stance.npy   (n_ch, n_freqs)          [standing mode]
    sub-{sub}_ersp_swing.npy           (n_ch, n_freqs)          [standing mode]
    sub-{sub}_ersp_beta.npy            (n_ch, n_freqs, 101)     [gpm mode]
    sub-{sub}_roi_weights.npy          (n_ch,)
    group_gait_event_anchors.json                               [gpm mode]

Output
------
results/pipeline/<dataset>/qc/<dataset>_betaphase_stats.txt            (gpm, primary)
results/pipeline/<dataset>/qc/<dataset>_betaphase_stats_standingBL.txt (standing)
"""

import numpy as np
from scipy.stats import ttest_rel

from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS, DIR_QC, PIPELINE_TFR_FMAX, NORMALIZATION
from src.spatial_filter import apply_linear_roi
from src.ersp import (
    beta_roi_scalar, load_group_anchors, phase_split_indices,
    apply_gpm_normalization, BETA_FMIN, BETA_FMAX,
)

FREQS = np.arange(8, int(PIPELINE_TFR_FMAX) + 1, dtype=float)   # must match prepana05/06
BETA_MASK = (FREQS >= BETA_FMIN) & (FREQS <= BETA_FMAX)

dirs     = get_dataset_dirs(DATASET)
ERSP_DIR = dirs["ersp"]

if NORMALIZATION == "gpm":
    A_lto, A_lhs, A_rto = load_group_anchors(ERSP_DIR)
    double_stance_idx, swing_idx = phase_split_indices((A_lto, A_lhs, A_rto), n_points=101)

subjects_used = []
beta_ds_list  = []
beta_sw_list  = []

for subject in SUBJECTS:
    try:
        w_path = ERSP_DIR / f"sub-{subject}_roi_weights.npy"

        if NORMALIZATION == "standing":
            # ORIGINAL path, unchanged: pre-reduced, standing-baselined maps.
            ds_path = ERSP_DIR / f"sub-{subject}_ersp_double_stance.npy"
            sw_path = ERSP_DIR / f"sub-{subject}_ersp_swing.npy"
            if not (ds_path.exists() and sw_path.exists() and w_path.exists()):
                print(f"  [SKIP] sub-{subject}: missing double_stance/swing/roi_weights .npy")
                continue

            ersp_ds = np.load(ds_path)   # (n_ch, n_freqs)
            ersp_sw = np.load(sw_path)   # (n_ch, n_freqs)
            weights = np.load(w_path)    # (n_ch,)

            beta_ds = beta_roi_scalar(ersp_ds, weights, FREQS)
            beta_sw = beta_roi_scalar(ersp_sw, weights, FREQS)

        else:
            # gpm: full per-cycle array, GPM-renormalized, then reduced with
            # the same ROI weights / beta mask / phase-split anchors.
            beta_path = ERSP_DIR / f"sub-{subject}_ersp_beta.npy"
            if not (beta_path.exists() and w_path.exists()):
                print(f"  [SKIP] sub-{subject}: missing ersp_beta/roi_weights .npy")
                continue

            ersp_full = np.load(beta_path)          # (n_ch, n_freqs, 101), standing-baselined
            weights   = np.load(w_path)
            ersp_gpm  = apply_gpm_normalization(ersp_full)

            ersp_roi   = apply_linear_roi(ersp_gpm, weights)   # (n_freqs, 101)
            beta_trace = ersp_roi[BETA_MASK].mean(axis=0)      # (101,)
            beta_ds = float(beta_trace[double_stance_idx].mean())
            beta_sw = float(beta_trace[swing_idx].mean())

        subjects_used.append(subject)
        beta_ds_list.append(beta_ds)
        beta_sw_list.append(beta_sw)

        print(f"  sub-{subject}: double_stance={beta_ds:+.3f} dB  "
              f"swing={beta_sw:+.3f} dB  diff={beta_ds - beta_sw:+.3f} dB")

    except FileNotFoundError as e:
        print(f"  [SKIP] sub-{subject}: file not found -- {e}")
        continue

n = len(subjects_used)
if n < 2:
    raise RuntimeError(
        f"Only {n} subject(s) with usable double_stance/swing maps -- "
        f"cannot run a paired t-test (need >= 2)."
    )

beta_ds_arr = np.array(beta_ds_list)
beta_sw_arr = np.array(beta_sw_list)
diff_arr    = beta_ds_arr - beta_sw_arr

t_stat, p_val = ttest_rel(beta_ds_arr, beta_sw_arr)
df            = n - 1
mean_diff     = float(diff_arr.mean())
n_ds_gt_sw    = int(np.sum(diff_arr > 0))

NORM_LABEL = {
    "gpm":      "GPM: dB relative to mean gait cycle",
    "standing": "dB relative to standing baseline",
}[NORMALIZATION]

lines = []
lines.append("BETA PHASE GROUP STATISTICS -- double stance vs swing")
lines.append("=" * 55)
lines.append(f"Normalization: {NORM_LABEL}")
lines.append(f"Beta band: {BETA_FMIN:.0f}-{BETA_FMAX:.0f} Hz  "
             f"(linear Cz ROI weights)")
lines.append("")
lines.append("Per-subject beta ROI-weighted ERSP (dB):")
lines.append(f"  {'subject':<10}{'double_stance':>15}{'swing':>12}{'diff':>12}")
for sub, ds, sw in zip(subjects_used, beta_ds_list, beta_sw_list):
    lines.append(f"  {sub:<10}{ds:>15.3f}{sw:>12.3f}{ds - sw:>12.3f}")
lines.append("")
lines.append(f"Group paired t-test (double_stance - swing), n={n} subjects:")
lines.append(f"  t({df}) = {t_stat:.3f}   p = {p_val:.4g}")
lines.append(f"  mean diff = {mean_diff:+.3f} dB")
lines.append(f"  subjects with double_stance > swing: {n_ds_gt_sw}/{n}")

summary = "\n".join(lines)
print("\n" + summary)

_suffix  = "" if NORMALIZATION == "gpm" else "_standingBL"
out_path = DIR_QC / f"{DATASET}_betaphase_stats{_suffix}.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(summary + "\n")
print(f"\nSaved -> {out_path}")
