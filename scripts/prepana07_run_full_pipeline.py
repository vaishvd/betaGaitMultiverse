"""
Master pipeline runner for betaGaitMultiverse.

Runs all six canonical pipeline stages in sequence for all subjects
defined in src/config.SUBJECTS.

Usage (interactive window):
    Set MODE at the bottom of this file, then run the file.
    MODE = "all"   — run all stages in sequence
    MODE = "from"  — run from FROM_STAGE to end
    MODE = "only"  — run one stage only (set ONLY_STAGE)

Each stage script is run as a subprocess so that MNE's memory is fully
released between stages. Stage failures are reported but do not stop
the remaining stages unless STRICT = True.

Stages:
    1  prepana01_prep_gaitevents.py    gait event detection from motion capture
    2  prepana02_raw2ica.py            preprocessing + ICA fitting
    3  prepana03_ica2clean.py          ICLabel + ICA application
    4  prepana04_clean2gaitcycles.py   gait segment extraction
    5  prepana05_gaitcycles2tfr.py     TFR + ERSP computation
    6  prepana06_plotbetagait.py       group beta ERSP figure
"""

import subprocess
import sys
import time
from pathlib import Path

from src.config import SUBJECTS, DATASET, DIR_SCRIPTS

STAGES = [
    (1, "prepana01_prep_gaitevents.py",  "gait event detection"),
    (2, "prepana02_raw2ica.py",          "preprocessing + ICA fit"),
    (3, "prepana03_ica2clean.py",        "ICLabel + ICA apply"),
    (4, "prepana04_clean2gaitcycles.py", "gait segment extraction"),
    (5, "prepana05_gaitcycles2tfr.py",   "TFR + ERSP"),
    (6, "prepana06_plotbetagait.py",     "group ERSP figure"),
]

def run_stage(stage_num, script_name, description, strict=False):
    script = DIR_SCRIPTS / script_name
    if not script.exists():
        print(f"\n[ERROR] Stage {stage_num} script not found: {script}")
        if strict:
            sys.exit(1)
        return False

    print(f"\n{'='*60}")
    print(f"Stage {stage_num}/6 — {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(DIR_SCRIPTS.parent)
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAILED] Stage {stage_num} exited with code "
              f"{result.returncode} after {elapsed:.1f}s")
        if strict:
            sys.exit(result.returncode)
        return False

    print(f"\n[DONE] Stage {stage_num} completed in {elapsed:.1f}s")
    return True


# ---------------------------------------------------------------------------
# Run — set MODE and options before executing in interactive window
#
#   "all"   : run all stages in sequence
#   "from"  : run from FROM_STAGE to end
#   "only"  : run one stage only
# ---------------------------------------------------------------------------
MODE        = "all"    # change this before running
FROM_STAGE  = 1        # only used when MODE = "from"
ONLY_STAGE  = 6        # only used when MODE = "only"
STRICT      = False    # if True, stop on first stage failure

if MODE == "all":
    stages_to_run = STAGES

elif MODE == "from":
    stages_to_run = [s for s in STAGES if s[0] >= FROM_STAGE]
    if not stages_to_run:
        print(f"No stages with number >= {FROM_STAGE}")
        stages_to_run = []

elif MODE == "only":
    stages_to_run = [s for s in STAGES if s[0] == ONLY_STAGE]
    if not stages_to_run:
        print(f"No stage with number {ONLY_STAGE}")
        stages_to_run = []

else:
    print(f"Unknown MODE '{MODE}'. Choose: all / from / only")
    stages_to_run = []

if stages_to_run:
    print(f"\nbetaGaitMultiverse — canonical pipeline")
    print(f"Dataset  : {DATASET}")
    print(f"Subjects : {SUBJECTS}")

    results  = []
    t_total  = time.time()

    for stage_num, script_name, description in stages_to_run:
        ok = run_stage(stage_num, script_name, description, STRICT)
        results.append((stage_num, description, ok))

    print(f"\n{'='*60}")
    print(f"Pipeline complete in {(time.time()-t_total)/60:.1f} min")
    print(f"{'='*60}")
    for stage_num, description, ok in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] Stage {stage_num}: {description}")

    failed = [r for r in results if not r[2]]
    if failed:
        print(f"\n  {len(failed)} stage(s) failed.")
