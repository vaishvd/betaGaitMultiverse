"""
Main pipeline to load and analyze EEG data.

Runs all seven canonical pipeline stages in sequence for all subjects
defined in src/config.SUBJECTS.

Individual stages are implemented in prepanaXX_*.py scripts, which arealled by this main script. Each stage is designed to be runnable
independently for ease of debugging and development, but this main script provides a single entry point to run the entire pipeline from start to finish.

"""

import subprocess
import sys
import time

from src.config import SUBJECTS, DATASET, DIR_SCRIPTS

STAGES = [
    (1, "prepana01_prep_gaitevents.py",     "gait event detection"),
    (2, "prepana02_raw2ica.py",             "preprocessing + ICA fit"),
    (3, "prepana03_ica2clean.py",           "ICLabel + ICA apply"),
    (4, "prepana04_clean2gaitcycles.py",    "gait segment extraction"),
    (5, "prepana05_gaitcycles2tfr.py",      "TFR + ERSP"),
    (6, "prepana06_plotbetagait.py",        "group ERSP figure"),
    (7, "prepana07_betaphase_stats.py",     "beta phase group t-test"),
]

print(f"Dataset  : {DATASET}")
print(f"Subjects : {SUBJECTS}")

results = []
t_total = time.time()

for stage_num, script_name, description in STAGES:
    script = DIR_SCRIPTS / script_name
    print(f"\n{'='*60}")
    print(f"Stage {stage_num}/7 — {description}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(DIR_SCRIPTS.parent)
    )
    elapsed = time.time() - t0
    ok = result.returncode == 0

    status = "DONE" if ok else f"FAILED (code {result.returncode})"
    print(f"\n[{status}] Stage {stage_num} in {elapsed:.1f}s")
    results.append((stage_num, description, ok))

print(f"\n{'='*60}")
print(f"Pipeline complete in {(time.time()-t_total)/60:.1f} min")
print(f"{'='*60}")
for stage_num, description, ok in results:
    print(f"  [{'OK  ' if ok else 'FAIL'}] Stage {stage_num}: {description}")