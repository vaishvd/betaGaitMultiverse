"""
Entry point for the betaGaitMultiverse multiverse analysis.

Usage:
    python scripts/multiverse/run_multiverse.py            # create + run + results
    python scripts/multiverse/run_multiverse.py --create   # generate universe scripts
    python scripts/multiverse/run_multiverse.py --run      # run all universes
    python scripts/multiverse/run_multiverse.py --results  # load and display results
    python scripts/multiverse/run_multiverse.py --universe 3  # run one universe

Outputs written to: results/multiverse/comet/beta_gait_multiverse/
"""

import argparse
import sys
import subprocess
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from comet.multiverse import Multiverse
from src.multiverse import MULTIVERSE_NAME, forking_paths, analysis_template
from src.config import DIR_MULTIVERSE, DIR_MULTIVERSE_OUTPUTS

MV_DIR = DIR_MULTIVERSE / "comet" / MULTIVERSE_NAME


def get_mv():
    MV_DIR.mkdir(parents=True, exist_ok=True)
    return Multiverse(name=MULTIVERSE_NAME, path=str(MV_DIR))


def cmd_create(mv):
    n = 1
    for v in forking_paths.values():
        n *= len(v)
    print(f"Creating {n} universe scripts ({len(forking_paths)} decisions)...")
    for k, v in forking_paths.items():
        print(f"  {k}: {v}")
    mv.create(analysis_template, forking_paths)
    scripts = list((MV_DIR / "scripts").glob("universe_*.py"))
    print(f"\nGenerated {len(scripts)} universe scripts -> {MV_DIR / 'scripts'}")
    mv.summary()


def cmd_run(mv, universe=None):
    if universe is not None:
        script = MV_DIR / "scripts" / f"universe_{universe}.py"
        if not script.exists():
            print(f"Error: {script} not found. Run --create first.")
            return
        print(f"Running universe {universe}: {script}")
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"Universe {universe} exited with code {result.returncode}")
    else:
        print("Running all universes sequentially...")
        mv.run(parallel=1)


def cmd_results(mv):
    try:
        df = mv.get_results(as_df=True)
    except Exception as e:
        print(f"Could not load results: {e}")
        return

    if df is None or len(df) == 0:
        print("No results found. Run the multiverse first.")
        return

    print(f"\n{'='*60}")
    print(f"MULTIVERSE RESULTS  ({len(df)} universes)")
    print(f"{'='*60}")

    # Specification curve — Simonsohn et al. 2020 format
    try:
        mv.specification_curve(measure="group_t_mean")
        print("Specification curve saved.")
    except Exception as e:
        print(f"Spec curve failed: {e}")

    decision_cols = [c for c in df.columns if c.startswith("Value")]
    outcome_cols  = ["group_t_mean", "group_t_std",
                     "group_test_p", "n_subjects_ok"]
    label_cols    = [c for c in df.columns if c.startswith("Decision")]

    display_cols = label_cols + decision_cols + \
                   [c for c in outcome_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))

    out = DIR_MULTIVERSE_OUTPUTS / "multiverse_outcomes.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"\nFull results saved -> {out}")

    # Summary: which decision has the largest effect on group_t_mean?
    if "group_t_mean" in df.columns and len(df) > 1:
        print(f"\n{'='*60}")
        print("Decision influence on group_t_mean:")
        for i, (k, vals) in enumerate(forking_paths.items(), 1):
            val_col = f"Value {i}"
            if val_col not in df.columns:
                continue
            group_means = df.groupby(val_col)["group_t_mean"].mean()
            spread = group_means.max() - group_means.min()
            print(f"  {k:>20}: range = {spread:.3f}  "
                  f"({group_means.to_dict()})")


parser = argparse.ArgumentParser()
parser.add_argument("--create",    action="store_true")
parser.add_argument("--run",       action="store_true")
parser.add_argument("--results",   action="store_true")
parser.add_argument("--universe",  type=int, default=None)
args = parser.parse_args()

mv = get_mv()

if args.create:
    cmd_create(mv)
elif args.run or args.universe is not None:
    cmd_run(mv, args.universe)
elif args.results:
    cmd_results(mv)
else:
    cmd_create(mv)
    cmd_run(mv)
    cmd_results(mv)
