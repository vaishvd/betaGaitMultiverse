"""
Master pipeline runner for betaGaitMultiverse.

Runs all six canonical pipeline stages in sequence for all subjects
defined in src/config.SUBJECTS.

Usage:
    python scripts/run_pipeline.py           # run all stages
    python scripts/run_pipeline.py --from 3  # start from stage 3
    python scripts/run_pipeline.py --only 5  # run only stage 5

Each stage script is run as a subprocess so that MNE's memory is fully
released between stages. Stage failures are reported but do not stop
the remaining stages unless --strict is passed.

Stages:
    1  prepana01_prep_gaitevents.py    gait event detection from motion capture
    2  prepana02_raw2ica.py            preprocessing + ICA fitting
    3  prepana03_ica2clean.py          ICLabel + ICA application
    4  prepana04_clean2gaitcycles.py   gait segment extraction
    5  prepana05_gaitcycles2tfr.py     TFR + ERSP computation
    6  prepana06_plotbetagait.py       group beta ERSP figure
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import SUBJECTS, DATASET

STAGES = [
    (1, "prepana01_prep_gaitevents.py",  "gait event detection"),
    (2, "prepana02_raw2ica.py",          "preprocessing + ICA fit"),
    (3, "prepana03_ica2clean.py",        "ICLabel + ICA apply"),
    (4, "prepana04_clean2gaitcycles.py", "gait segment extraction"),
    (5, "prepana05_gaitcycles2tfr.py",   "TFR + ERSP"),
    (6, "prepana06_plotbetagait.py",     "group ERSP figure"),
]

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_stage(stage_num, script_name, description, strict=False):
    script = SCRIPTS_DIR / script_name
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
        cwd=str(repo_root)
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


def main():
    parser = argparse.ArgumentParser(
        description="Run the betaGaitMultiverse canonical pipeline"
    )
    parser.add_argument(
        "--from", dest="from_stage", type=int, default=1,
        help="Start from this stage number (default: 1)"
    )
    parser.add_argument(
        "--only", type=int, default=None,
        help="Run only this stage number"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Stop on first stage failure"
    )
    args = parser.parse_args()

    print(f"\nbetaGaitMultiverse — canonical pipeline")
    print(f"Dataset  : {DATASET}")
    print(f"Subjects : {SUBJECTS}")

    stages_to_run = STAGES
    if args.only is not None:
        stages_to_run = [s for s in STAGES if s[0] == args.only]
        if not stages_to_run:
            print(f"No stage with number {args.only}")
            sys.exit(1)
    else:
        stages_to_run = [s for s in STAGES if s[0] >= args.from_stage]

    results = []
    t_total = time.time()
    for stage_num, script_name, description in stages_to_run:
        ok = run_stage(stage_num, script_name, description, args.strict)
        results.append((stage_num, description, ok))

    print(f"\n{'='*60}")
    print(f"Pipeline complete in {(time.time()-t_total)/60:.1f} min")
    print(f"{'='*60}")
    for stage_num, description, ok in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] Stage {stage_num}: {description}")

    failed = [r for r in results if not r[2]]
    if failed:
        sys.exit(1)
