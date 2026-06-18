"""
betaGaitMultiverse — multiverse analysis entry point.

Defines all preprocessing decision nodes and runs the full multiverse.
Forking paths are defined in this file (forking_paths dict below). The
analysis_template function defines the analysis to be run for each universe, and is imported by COMET's multiverse runner.

Decision nodes and literature basis:
    use_asr       : Mullen et al. 2015 IEEE TBME; Gorjan et al. 2022 J Neural Eng
    brain_thresh  : Pion-Tonachini et al. 2019 NeuroImage
    highpass_hz   : Tran et al. 2020 J Neurosci Methods
    lowpass_hz    : affects ICA quality and muscle artifact retention
    baseline_type : Makeig et al. 1993; Seeber et al. 2015 J Neurosci
    phase_window  : "full" = entire stance/swing phase;
                    "peak" = literature-defined peak windows only
                    (stance 5-15%, swing 75-85%)
                    Petersen et al. 2012 Eur J Neurosci;
                    Bulea et al. 2015 Front Hum Neurosci
"""

from comet.multiverse import Multiverse


# ── Forking paths ─────────────────────────────────────────────────────────
forking_paths = {
    "use_asr":       [False, True],
    "brain_thresh":  [0.7, 0.9],
    "baseline_type": ["standing", "walking_mean"],
    "phase_window":  ["full", "peak"],
}

# ── Analysis template ──────────────────────────────────────────────────────
def analysis_template():
    import comet
    import numpy as np
    from src.config import MULTIVERSE_SUBJECTS as SUBJECTS
    from src.multiverse_pipeline import run_subject_multiverse
    decisions = {
        "use_asr":       {{ use_asr }},
        "brain_thresh":  {{ brain_thresh }},
        "highpass_hz":   1.0,
        "lowpass_hz":    60.0,
        "baseline_type": {{ baseline_type }},
        "phase_window":  {{ phase_window }},
    }
    t_stats, n_ok = [], 0
    for subj in SUBJECTS:
        try:
            result = run_subject_multiverse(subj, decisions)
        except Exception as e:
            print(f"  sub-{subj}: FAILED - {e}")
            continue
        t_stats.append(result["t_stat"])
        n_ok += 1
    group_t_mean = float(np.mean(t_stats)) if t_stats else float("nan")
    comet.utils.save_universe_results({
        "group_t_mean":  group_t_mean,
        "n_subjects_ok": n_ok,
    })
    
# ── Create universes ──────────────────────────────────────────────────────
mverse = Multiverse(name="beta_gait_multiverse")
mverse.create(analysis_template, forking_paths)
mverse.summary()
