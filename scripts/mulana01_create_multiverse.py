"""
betaGaitMultiverse — create multiverse universes.

Defines all preprocessing decision nodes and generates the full set of
universe scripts via COMET. Run this once before mulana02_run_multiverse.py.

Decision nodes and literature basis:
    use_asr       : artifact subspace reconstruction on/off
                    Mullen et al. 2015; Gorjan et al. 2022
    use_gedai     : GEDAI denoising on/off
                    neurotuning.github.io/gedai
    baseline_type : standing vs walking_mean
                    Makeig et al. 1993; Seeber et al. 2015
    brain_thresh  : ICLabel threshold 0.7 vs 0.9
                    Pion-Tonachini et al. 2019
"""

import shutil
from pathlib import Path

from comet.multiverse import Multiverse
from src.config import MULTIVERSE_NAME, DIR_MULTIVERSE, DIR_MULTIVERSE_OUTPUTS, DIR_PROJ

# --- Clear old COMET state, PKLs, and universe scripts ---
for pattern in ("*.pkl", "*.json"):
    for f in DIR_MULTIVERSE.glob(pattern):
        f.unlink()
        print(f"  Removed: {f.name}")

universes_dir = DIR_MULTIVERSE / "universes"
if universes_dir.exists():
    shutil.rmtree(universes_dir)
    print(f"  Removed: universes/")

for f in DIR_MULTIVERSE_OUTPUTS.glob("*.pkl"):
    f.unlink()
    print(f"  Removed outputs: {f.name}")

for d in DIR_PROJ.rglob(MULTIVERSE_NAME):
    if d.is_dir():
        shutil.rmtree(d)
        print(f"  Removed: {d.relative_to(DIR_PROJ)}")

for s in DIR_PROJ.rglob("universe_*.py"):
    s.unlink()
    print(f"  Removed: {s.relative_to(DIR_PROJ)}")

print("  Old COMET state cleared.\n")

# --- Forking paths ---
forking_paths = {
    "use_asr":      [False, True],
    "use_gedai":    [False, True],
    "baseline_type":["standing", "walking_mean"],
    "brain_thresh": [0.7, 0.9],
}


# --- Analysis template ---
def analysis_template():
    import comet
    import numpy as np
    from src.config import MULTIVERSE_SUBJECTS as SUBJECTS
    from src.multiverse_pipeline import run_subject_multiverse
    decisions = {
        "use_asr":       {{ use_asr }},
        "use_gedai":     {{ use_gedai }},
        "brain_thresh":  {{ brain_thresh }},
        "highpass_hz":   1.0,
        "lowpass_hz":    40.0,
        "baseline_type": {{ baseline_type }},
        "phase_window":  "segment",
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
        "t_stats":       t_stats,        # list of per-subject t-stats for CI
        "n_subjects_ok": n_ok,
    })


# --- Create universes ---
mverse = Multiverse(name=MULTIVERSE_NAME)
mverse.create(analysis_template, forking_paths)
mverse.summary()
