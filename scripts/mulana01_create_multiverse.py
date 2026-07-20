"""
betaGaitMultiverse — create multiverse universes.

Defines all preprocessing decision nodes and generates the full set of
universe scripts via COMET. Run this once before mulana02_run_multiverse.py.

Decision nodes and literature basis:
    highpass_hz  : FIR high-pass cutoff (Hz) applied before ICA fit
                   0.5 / 1.0 / 2.0
    asr_mode     : artifact subspace reconstruction burst cutoff (SD)
                   "off" (no ASR) / "sd3" (aggressive) / "sd20" (lenient)
                   Mullen et al. 2015; Gorjan et al. 2022
    iclabel_rule : ICLabel component-exclusion rule, applied to one
                   shared ICA fit per (highpass_hz, asr_mode) branch --
                   does not refit ICA
                   "conservative" (P(brain) > 0.9) /
                   "balanced" (P(brain) > 0.7) /
                   "liberal" (reject only P(muscle) > 0.9 or P(eye) > 0.9)
                   Pion-Tonachini et al. 2019

3 x 3 x 3 = 27 universes total. Each universe computes the same
event-anchored double-stance-vs-swing beta contrast as the canonical
pipeline (see src/ersp.py, src/multiverse_pipeline.py).
"""

import shutil

from comet.multiverse import Multiverse
from src.config import (
    MULTIVERSE_NAME, DIR_MULTIVERSE, DIR_MULTIVERSE_OUTPUTS,
    DIR_MULTIVERSE_COMET, DIR_PROJ,
)

# --- Clear old COMET state, PKLs, and universe scripts ---
# Scoped entirely to this ACTIVE_DATASET's own directories -- unlike an
# earlier version of this cleanup, this never searches or deletes
# anything outside DIR_MULTIVERSE (results/multiverse/<dataset>/), so
# re-creating one dataset's universes can never touch the other
# dataset's cached branches, COMET state, or outputs.
for pattern in ("*.pkl", "*.json"):
    for f in DIR_MULTIVERSE.glob(pattern):
        f.unlink()
        print(f"  Removed: {f.relative_to(DIR_PROJ)}")

for f in DIR_MULTIVERSE_OUTPUTS.glob("*.pkl"):
    f.unlink()
    print(f"  Removed: {f.relative_to(DIR_PROJ)}")

if DIR_MULTIVERSE_COMET.exists():
    shutil.rmtree(DIR_MULTIVERSE_COMET)
    print(f"  Removed: {DIR_MULTIVERSE_COMET.relative_to(DIR_PROJ)}/")

print("  Old COMET state cleared.\n")

# --- Forking paths ---
forking_paths = {
    "highpass_hz":  [0.5, 1.0, 2.0],
    "asr_mode":     ["off", "sd3", "sd20"],
    "iclabel_rule": ["conservative", "balanced", "liberal"],
}


# --- Analysis template ---
def analysis_template():
    import comet
    import numpy as np
    from scipy.stats import ttest_rel
    from src.config import MULTIVERSE_SUBJECTS as SUBJECTS
    from src.multiverse_pipeline import run_subject_multiverse
    # NOTE: comet's Jinja substitution already double-quotes string fork
    # values (see comet.multiverse.Multiverse._render_val) -- do not wrap
    # asr_mode / iclabel_rule in manual quotes here, that would double-quote.
    decisions = {
        "highpass_hz":  {{ highpass_hz }},
        "asr_mode":     {{ asr_mode }},
        "iclabel_rule": {{ iclabel_rule }},
        "lowpass_hz":   40.0,
    }
    beta_diffs, beta_ds_list, beta_sw_list, n_ok = [], [], [], 0
    for subj in SUBJECTS:
        try:
            result = run_subject_multiverse(subj, decisions)
        except Exception as e:
            print(f"  sub-{subj}: FAILED - {e}")
            continue
        beta_diffs.append(result["beta_diff"])
        beta_ds_list.append(result["beta_double_stance"])
        beta_sw_list.append(result["beta_swing"])
        n_ok += 1

    # Group paired t-test (double_stance vs swing across subjects),
    # computed the same way as prepana07_betaphase_stats.py.
    if n_ok >= 2:
        t_stat, _ = ttest_rel(np.array(beta_ds_list), np.array(beta_sw_list))
        group_t_mean = float(t_stat)
    else:
        group_t_mean = float("nan")

    comet.utils.save_universe_results({
        "group_t_mean":  group_t_mean,   # single group-level paired t (prepana07 method)
        "t_stats":       beta_diffs,      # list of per-subject beta contrasts (dB) for CI
        "n_subjects_ok": n_ok,
    })


# --- Create universes ---
# path= pins COMET's working directory to this dataset's own nested
# folder (results/multiverse/<dataset>/comet/) instead of its default
# (the calling script's directory) -- see src.config.DIR_MULTIVERSE_COMET.
mverse = Multiverse(name=MULTIVERSE_NAME, path=str(DIR_MULTIVERSE_COMET))
mverse.create(analysis_template, forking_paths)
mverse.summary()
