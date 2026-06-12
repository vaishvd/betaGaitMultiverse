"""
Multiverse specification for betaGaitMultiverse.

FORKING PATHS — defined once here, used by COMET to generate all
universe combinations. Add new decision nodes here only.

Current decisions (32 universes):
  use_asr        : bool   — apply ASR before ICA
  brain_thresh   : float  — ICLabel brain probability threshold
  highpass_hz    : float  — high-pass filter cutoff in Hz
  lowpass_hz     : int|None — low-pass filter cutoff (None = skip)
  baseline_type  : str    — ERSP baseline source

Primary outcome: mean t-statistic (stance vs swing beta, two-tailed,
across gait cycles within subject) averaged across subjects.

Literature basis for each node:
  use_asr       : Mullen 2015 IEEE TBME; Gorjan 2022 J Neural Eng
  brain_thresh  : Pion-Tonachini 2019 NeuroImage
  highpass_hz   : Tran 2020 J Neurosci Methods; Luck 2014
  lowpass_hz    : affects ICA quality and muscle artifact retention
  baseline_type : Makeig 1993; Seeber 2015 J Neurosci

Usage:
    python scripts/run_multiverse.py
"""

from comet.multiverse import Multiverse

MULTIVERSE_NAME = "beta_gait_multiverse"

forking_paths = {
    "use_asr":       [False, True],
    "brain_thresh":  [0.7, 0.9],
    "highpass_hz":   [0.1, 2.0],
    "lowpass_hz":    [40, None],
    "baseline_type": ["standing", "walking_mean"],
}


def analysis_template():
    # --- imports (each universe script is a standalone subprocess) ---
    import sys
    import os
    import pickle
    import numpy as np
    from scipy.stats import ttest_1samp

    # Universe scripts live at:
    #   {project_root}/results/multiverse/beta_gait_multiverse/scripts/universe_N.py
    # So 4 levels up from __file__ reaches {project_root}.
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.multiverse_pipeline import run_subject_multiverse
    from src.config import DATASET, MULTIVERSE_SUBJECTS as SUBJECTS

    # --- Universe identity: derived from the script filename (universe_N.py) ---
    universe_id = int(
        os.path.basename(__file__).split("_")[-1].split(".")[0]
    )
    decisions = {
        "use_asr":       {{ use_asr }},
        "brain_thresh":  {{ brain_thresh }},
        "highpass_hz":   {{ highpass_hz }},
        "lowpass_hz":    {{ lowpass_hz }},
        "baseline_type": {{ baseline_type }},
    }

    print(f"\n=== Universe {universe_id} ===")
    print(f"  Decisions: {decisions}")

    # --- Run pipeline for every subject ---
    results_all = []
    for subject in SUBJECTS:
        try:
            res = run_subject_multiverse(subject, DATASET, decisions)
            res["universe_id"] = universe_id
            results_all.append(res)
            print(f"  sub-{subject} done: "
                  f"t={res['t_stat']:+.2f}  "
                  f"stance={res['beta_stance_mean']:+.2f}  "
                  f"swing={res['beta_swing_mean']:+.2f}  "
                  f"n={res['n_cycles']}")
        except Exception as e:
            import traceback
            print(f"  sub-{subject} FAILED: {e}")
            traceback.print_exc()
            results_all.append({
                "subject":          subject,
                "universe_id":      universe_id,
                "t_stat":           float("nan"),
                "t_pval":           float("nan"),
                "beta_stance_mean": float("nan"),
                "beta_swing_mean":  float("nan"),
                "n_cycles":         0,
                "n_brain_ics":      0,
                "baseline_ok":      False,
                "error":            str(e),
                **decisions,
            })

    # --- Group outcome: mean t-statistic across subjects ---
    valid   = [r for r in results_all
               if not np.isnan(r.get("t_stat", float("nan")))]
    t_stats = [r["t_stat"] for r in valid]

    group_t_mean = float(np.mean(t_stats)) if t_stats else float("nan")
    group_t_std  = float(np.std(t_stats))  if t_stats else float("nan")

    # One-sample t-test: is the group mean t-stat different from zero?
    if len(t_stats) >= 2:
        group_test_t, group_test_p = ttest_1samp(t_stats, 0)
        group_test_t = float(group_test_t)
        group_test_p = float(group_test_p)
    else:
        group_test_t = group_test_p = float("nan")

    # --- Decision record (COMET format) ---
    __decisions = {}
    for i, (k, v) in enumerate(decisions.items(), 1):
        __decisions[f"Decision {i}"] = k
        __decisions[f"Value {i}"]    = str(v)

    group_result = {
        "__decisions":    __decisions,
        "universe_id":    universe_id,
        "decisions":      decisions,
        "n_subjects_ok":  len(valid),
        "group_t_mean":   group_t_mean,
        "group_t_std":    group_t_std,
        "group_test_t":   group_test_t,
        "group_test_p":   group_test_p,
        "per_subject":    results_all,
    }

    # --- Save (COMET _combine_results reads from scripts/temp/universe_N.pkl) ---
    out_dir  = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"universe_{universe_id}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(group_result, f)

    print(f"  Group t mean: {group_t_mean:.2f} +/- {group_t_std:.2f}")
    print(f"  Universe {universe_id} saved -> {out_path}")
