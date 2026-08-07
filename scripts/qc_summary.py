"""
Aggregate per-subject QC records into a group summary TSV.

Run after all subjects have been processed:
    python scripts/qc_summary.py

Reads all *_qc.json files from the active dataset's QC directory
(src.config.DIR_QC, results/pipeline/<dataset>/qc/) and writes, into that
same per-dataset directory:
    <dataset>_qc_summary.tsv   -- one row per subject per stage
    <dataset>_qc_flags.tsv     -- one row per subject, one column per stage flag
"""

from pathlib import Path
import pandas as pd
from src.paths import get_dataset_dirs
from src.qc import write_qc_summary

from src.config import DATASET, SUBJECTS, DIR_QC

dirs = get_dataset_dirs(DATASET)
QC_DIR = dirs["qc"]

# Full summary
summary_path = DIR_QC / f"{DATASET}_qc_summary.tsv"
df = write_qc_summary(QC_DIR, summary_path)

if df.empty:
    print("No QC records found. Run the pipeline first.")
else:
    # Pivot to wide format: one row per subject, flag per stage as columns
    flags = df.pivot_table(
        index="subject",
        columns="stage",
        values="flag",
        aggfunc="first"
    ).reset_index()
    flags_path = DIR_QC / f"{DATASET}_qc_flags.tsv"
    flags.to_csv(flags_path, sep="\t", index=False)
    print(f"  Flag table -> {flags_path}")

    # Print subjects with any warn or fail
    problem_subjects = df[df["flag"] != "pass"]["subject"].unique()
    if len(problem_subjects):
        print(f"\n  Subjects with warn/fail flags: {sorted(problem_subjects)}")
    else:
        print("\n  All subjects passed all QC stages.")
