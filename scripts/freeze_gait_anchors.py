"""
freeze_gait_anchors.py
=======================
Deliberately (re)compute the group-median gait-event anchors for the
active dataset, pooled across every subject with a kept-cycles file in
d04_gaitepochs, and write them to the FROZEN calibration file
(d05_ersp/group_gait_event_anchors_frozen.json).

This is the ONLY place these anchors are computed. Both the reference
pipeline (prepana05_gaitcycles2tfr.py) and the multiverse pipeline
(multiverse_pipeline.py) read this frozen file via src.ersp.
load_group_anchors() -- neither one computes or overwrites it anymore.
Before 2026-08-07, prepana05 silently recomputed and overwrote this file
on every run, pooled over whatever subjects/settings happened to be
active that run; a same-day drift between a cached multiverse result and
a fresh reference-pipeline re-run (different anchors, not different
code) is what surfaced this as a real bug -- see the ASR=20 pre-flight
task's multiverse cross-check.

Run this BY HAND, deliberately, only when the anchors genuinely need
recomputing (e.g. a materially different subject cohort or
preprocessing chain) -- never as an automatic side effect of a pipeline
run. Prints the new values against the existing frozen file (if any)
before overwriting, so a re-run always shows exactly what changed.

Usage
-----
    BETAGAIT_DATASET=stepup   python scripts/freeze_gait_anchors.py
    BETAGAIT_DATASET=jacobsen python scripts/freeze_gait_anchors.py
"""
import json
from datetime import date

import pandas as pd
import numpy as np

from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS

dirs = get_dataset_dirs(DATASET)
GAITEPOCH_DIR = dirs["gaitepochs"]
ERSP_DIR      = dirs["ersp"]
FROZEN_PATH   = ERSP_DIR / "group_gait_event_anchors_frozen.json"

_f_lto, _f_lhs, _f_rto = [], [], []
subjects_pooled = []
for subject in SUBJECTS:
    meta_path = GAITEPOCH_DIR / f"sub-{subject}_cycles_kept.tsv"
    if not meta_path.exists():
        print(f"  sub-{subject}: no cycles_kept.tsv -- skipping")
        continue
    meta = pd.read_csv(meta_path, sep="\t")
    if len(meta) == 0:
        print(f"  sub-{subject}: cycles_kept.tsv is empty -- skipping")
        continue
    dur = meta["rhs_end_s"] - meta["rhs_start_s"]
    _f_lto.append(((meta["lto_s"] - meta["rhs_start_s"]) / dur).values)
    _f_lhs.append(((meta["lhs_s"] - meta["rhs_start_s"]) / dur).values)
    _f_rto.append(((meta["rto_s"] - meta["rhs_start_s"]) / dur).values)
    subjects_pooled.append(subject)

if not _f_lto:
    raise RuntimeError(
        "No sub-*_cycles_kept.tsv files found -- cannot compute group "
        "gait-event anchors. Run prepana01-04 first."
    )

A_lto = float(np.median(np.concatenate(_f_lto))) * 100
A_lhs = float(np.median(np.concatenate(_f_lhs))) * 100
A_rto = float(np.median(np.concatenate(_f_rto))) * 100
n_subjects_pooled = len(subjects_pooled)
n_cycles_pooled = sum(len(a) for a in _f_lto)

assert 0 < A_lto < A_lhs < A_rto < 100, (
    f"Group anchors are not monotonic: A_lto={A_lto}, A_lhs={A_lhs}, A_rto={A_rto}"
)

print(f"New anchors (pooled across {n_subjects_pooled} subjects, {n_cycles_pooled} cycles):")
print(f"  A_lto={A_lto:.4f}%  A_lhs={A_lhs:.4f}%  A_rto={A_rto:.4f}%")
print(f"  subjects: {subjects_pooled}")

if FROZEN_PATH.exists():
    with open(FROZEN_PATH) as f:
        old = json.load(f)
    print(f"\nExisting frozen file ({FROZEN_PATH.name}):")
    print(f"  A_lto={old['A_lto_pct']:.4f}%  A_lhs={old['A_lhs_pct']:.4f}%  A_rto={old['A_rto_pct']:.4f}%  "
          f"n={old['n_subjects_pooled']}  frozen_date={old.get('frozen_date', '?')}")
    print(f"  Diff: dA_lto={A_lto - old['A_lto_pct']:+.4f}  dA_lhs={A_lhs - old['A_lhs_pct']:+.4f}  "
          f"dA_rto={A_rto - old['A_rto_pct']:+.4f}")
    resp = input(f"\nOverwrite {FROZEN_PATH} with the new values above? [y/N] ")
    if resp.strip().lower() != "y":
        print("Not overwritten. Exiting.")
        raise SystemExit(0)
else:
    print(f"\nNo existing frozen file at {FROZEN_PATH} -- creating it.")

frozen = {
    "A_lto_pct": A_lto,
    "A_lhs_pct": A_lhs,
    "A_rto_pct": A_rto,
    "n_subjects_pooled": n_subjects_pooled,
    "n_cycles_pooled": n_cycles_pooled,
    "subjects_pooled": subjects_pooled,
    "frozen_date": date.today().isoformat(),
    "note": (
        "FROZEN calibration input. Computed once from the reference "
        "pipeline's kept gait cycles, pooled across every subject listed "
        "in subjects_pooled. This file is READ-ONLY in practice: no "
        "pipeline code path overwrites it (src.ersp.load_group_anchors "
        "raises if it's missing rather than recomputing it). If these "
        "anchors ever need recomputing, rerun this script (scripts/"
        "freeze_gait_anchors.py) by hand -- it will show the diff "
        "against the current frozen values before overwriting."
    ),
}
with open(FROZEN_PATH, "w") as f:
    json.dump(frozen, f, indent=2)
print(f"\nSaved -> {FROZEN_PATH}")
