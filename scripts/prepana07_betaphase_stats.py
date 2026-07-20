"""
ana07_betaphase_stats.py
=========================
Pipeline step 7: per-subject ERSP maps -> group beta double-stance-vs-swing
paired t-test.

Does not recompute TFR/ERSP -- consumes the per-subject dB maps already
written by prepana05_gaitcycles2tfr.py. Reduces each (n_ch, n_freqs) map
to a single beta-band (13-30 Hz) scalar per subject using the same linear
Cz ROI weights as prepana05/06, then runs a paired t-test across subjects
on the double-stance-minus-swing difference. This is the canonical
group-level inferential test for the beta double-stance-vs-swing contrast.

Input
-----
d05_ersp/
    sub-{sub}_ersp_double_stance.npy   (n_ch, n_freqs)
    sub-{sub}_ersp_swing.npy           (n_ch, n_freqs)
    sub-{sub}_roi_weights.npy          (n_ch,)

Output
------
results/pipeline/qc/betaphase_group_stats.txt
"""

import numpy as np
from scipy.stats import ttest_rel

from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS, DIR_QC
from src.ersp import beta_roi_scalar, BETA_FMIN, BETA_FMAX

FREQS = np.arange(8, 41, dtype=float)   # must match prepana05/06

dirs     = get_dataset_dirs(DATASET)
ERSP_DIR = dirs["ersp"]

subjects_used = []
beta_ds_list  = []
beta_sw_list  = []

for subject in SUBJECTS:
    try:
        ds_path = ERSP_DIR / f"sub-{subject}_ersp_double_stance.npy"
        sw_path = ERSP_DIR / f"sub-{subject}_ersp_swing.npy"
        w_path  = ERSP_DIR / f"sub-{subject}_roi_weights.npy"

        if not (ds_path.exists() and sw_path.exists() and w_path.exists()):
            print(f"  [SKIP] sub-{subject}: missing double_stance/swing/roi_weights .npy")
            continue

        ersp_ds = np.load(ds_path)   # (n_ch, n_freqs)
        ersp_sw = np.load(sw_path)   # (n_ch, n_freqs)
        weights = np.load(w_path)    # (n_ch,)

        beta_ds = beta_roi_scalar(ersp_ds, weights, FREQS)
        beta_sw = beta_roi_scalar(ersp_sw, weights, FREQS)

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

lines = []
lines.append("BETA PHASE GROUP STATISTICS -- double stance vs swing")
lines.append("=" * 55)
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

out_path = DIR_QC / "betaphase_group_stats.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(summary + "\n")
print(f"\nSaved -> {out_path}")
