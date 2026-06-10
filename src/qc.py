"""
Per-subject QC logging for the betaGaitMultiverse pipeline.

Each pipeline stage calls log_qc() to record scalar metrics and a
pass/warn/fail flag. After all subjects are processed, write_qc_summary()
writes a TSV with one row per subject per stage.

Flag definitions:
    pass  — all metrics within expected range
    warn  — one or more metrics at boundary; results usable but inspect
    fail  — critical metric out of range; subject should be excluded

Usage:
    from src.qc import log_qc, write_qc_summary
    log_qc(qc_dir, subject, stage, flag, metrics)
    write_qc_summary(qc_dir, out_path)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path


class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars to Python-native types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def log_qc(
    qc_dir: Path,
    subject: str,
    stage: str,
    flag: str,
    metrics: dict,
) -> None:
    """
    Write a JSON QC record for one subject and pipeline stage.

    Parameters
    ----------
    qc_dir  : directory where QC files are written (created if absent)
    subject : subject identifier string, e.g. 'S1'
    stage   : pipeline stage label, e.g. 'gait_events', 'preprocessing',
              'ica', 'gait_epochs', 'ersp'
    flag    : 'pass', 'warn', or 'fail'
    metrics : dict of scalar values to record, e.g.
              {'n_cycles': 302, 'mean_duration_s': 0.992}
    """
    assert flag in ("pass", "warn", "fail"), \
        f"flag must be 'pass', 'warn', or 'fail', got '{flag}'"
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    record = {"subject": subject, "stage": stage, "flag": flag, **metrics}
    out = qc_dir / f"sub-{subject}_{stage}_qc.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2, cls=_NumpyEncoder)


def write_qc_summary(qc_dir: Path, out_path: Path) -> pd.DataFrame:
    """
    Aggregate all per-subject JSON QC records into a TSV summary table.

    Reads every *_qc.json file in qc_dir, concatenates into a DataFrame,
    writes to out_path as TSV, and returns the DataFrame.
    Prints a flag count summary (pass/warn/fail per stage) to stdout.
    """
    qc_dir = Path(qc_dir)
    records = []
    for f in sorted(qc_dir.glob("*_qc.json")):
        with open(f) as fh:
            records.append(json.load(fh))
    if not records:
        print("  No QC records found.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Sort by stage order then subject
    stage_order = ["gait_events", "preprocessing", "ica",
                   "gait_epochs", "ersp"]
    df["stage_rank"] = df["stage"].apply(
        lambda s: stage_order.index(s) if s in stage_order else 99
    )
    df = df.sort_values(["stage_rank", "subject"]).drop(
        columns="stage_rank"
    ).reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  QC summary -> {out_path}")
    print(f"  Rows: {len(df)}")
    for stage in stage_order:
        sub = df[df["stage"] == stage]
        if len(sub):
            counts = sub["flag"].value_counts().to_dict()
            print(f"  {stage:>15}: "
                  f"pass={counts.get('pass',0)}  "
                  f"warn={counts.get('warn',0)}  "
                  f"fail={counts.get('fail',0)}")
    return df
