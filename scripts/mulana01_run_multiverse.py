"""
betaGaitMultiverse — multiverse analysis entry point.

Defines all preprocessing decision nodes and runs the full multiverse.

Usage (interactive window):
    Set MODE at the bottom of this file, then run the file.
    MODE = "create"   — generate universe scripts only
    MODE = "run"      — run all universes (sequential)
    MODE = "universe" — run one universe (set UNIVERSE = N)
    MODE = "results"  — load results and generate plots
    MODE = "all"      — create + run + results

Decision nodes and literature basis:
    use_asr       : Mullen et al. 2015 IEEE TBME; Gorjan et al. 2022 J Neural Eng
    brain_thresh  : Pion-Tonachini et al. 2019 NeuroImage
    highpass_hz   : Tran et al. 2020 J Neurosci Methods
    lowpass_hz    : affects ICA quality and muscle artifact retention
    baseline_type : Makeig et al. 1993; Seeber et al. 2015 J Neurosci
"""

from comet.multiverse import Multiverse
from src.config import DIR_MULTIVERSE, DIR_MULTIVERSE_OUTPUTS
from src.multiverse import MULTIVERSE_NAME, forking_paths


# ---------------------------------------------------------------------------
# Analysis template
# ---------------------------------------------------------------------------
def analysis_template():
    import comet
    import numpy as np
    import pandas as pd
    from scipy.stats import ttest_rel
    import os

    from src.config import DATASET, MULTIVERSE_SUBJECTS as SUBJECTS
    from src.multiverse_pipeline import run_subject_multiverse

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

    results_all = []
    for subject in SUBJECTS:
        try:
            res = run_subject_multiverse(subject, DATASET, decisions)
            res["universe_id"] = universe_id
            results_all.append(res)
            print(f"  sub-{subject}: t={res['t_stat']:+.2f}  "
                  f"stance={res['beta_stance_mean']:+.2f}  "
                  f"swing={res['beta_swing_mean']:+.2f}")
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

    valid   = [r for r in results_all
               if not np.isnan(r.get("t_stat", float("nan")))]
    t_stats = [r["t_stat"] for r in valid]

    group_t_mean = float(np.mean(t_stats)) if t_stats else float("nan")
    group_t_std  = float(np.std(t_stats))  if t_stats else float("nan")

    if len(t_stats) >= 2:
        from scipy.stats import ttest_1samp
        group_test_t, group_test_p = ttest_1samp(t_stats, 0)
    else:
        group_test_t = group_test_p = float("nan")

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
        "group_test_t":   float(group_test_t),
        "group_test_p":   float(group_test_p),
        "per_subject":    results_all,
    }

    print(f"  Group t mean: {group_t_mean:.2f} +/- {group_t_std:.2f}")
    comet.utils.save_universe_results(group_result)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

MV_DIR = DIR_MULTIVERSE / "comet" / MULTIVERSE_NAME
mverse = Multiverse(name=MULTIVERSE_NAME, path=str(MV_DIR))

# ---------------------------------------------------------------------------
# Run — set MODE and UNIVERSE before executing in interactive window
#
#   "create"   : generate universe scripts only
#   "run"      : run all universes (sequential)
#   "universe" : run a single universe (set UNIVERSE below)
#   "results"  : load results and generate plots
#   "all"      : create + run + results
# ---------------------------------------------------------------------------
MODE     = "results"   
UNIVERSE = 1          # only used when MODE = "universe"

if MODE == "create":
    mverse.create(analysis_template, forking_paths)
    mverse.summary()

elif MODE == "run":
    mverse.run(parallel=1)

elif MODE == "universe":
    mverse.run(universe=UNIVERSE)

elif MODE == "results":
    df = mverse.get_results(as_df=True)
    if df is None or len(df) == 0:
        print("No results found. Set MODE='run' first.")
    else:
        # Save outcomes TSV
        out_tsv = DIR_MULTIVERSE_OUTPUTS / "multiverse_outcomes.tsv"
        df.to_csv(out_tsv, sep="\t", index=False)
        print(f"Results saved -> {out_tsv}")
        print(f"Universes: {len(df)}  |  "
              f"t-mean range: {df['group_t_mean'].min():.2f} "
              f"to {df['group_t_mean'].max():.2f}")

        comet_results_dir = MV_DIR / "results"

        # Human-readable labels for decision axes
        name_map = {
            "use_asr":       "Artifact subspace reconstruction",
            "brain_thresh":  "ICLabel brain threshold",
            "highpass_hz":   "High-pass filter (Hz)",
            "lowpass_hz":    "Low-pass filter (Hz)",
            "baseline_type": "ERSP baseline",
        }

        # Per-subject CI bounds for specification curve
        import ast as _ast
        import scipy.stats as _stats

        def _compute_ci_bounds(cell, confidence=0.95):
            try:
                rows = _ast.literal_eval(str(cell))
                t_vals = [
                    float(r["t_stat"])
                    for r in rows
                    if isinstance(r.get("t_stat"), (int, float))
                    and not (r["t_stat"] != r["t_stat"])
                ]
                if len(t_vals) >= 2:
                    mean  = float(_stats.tmean(t_vals))
                    se    = float(_stats.tsem(t_vals))
                    df_n  = len(t_vals) - 1
                    h     = se * _stats.t.ppf((1 + confidence) / 2, df_n)
                    return (mean - h, mean + h)
                elif len(t_vals) == 1:
                    return (t_vals[0], t_vals[0])
                else:
                    return (float("nan"), float("nan"))
            except Exception:
                return (float("nan"), float("nan"))

        df["t_ci_bounds"] = df["per_subject"].apply(_compute_ci_bounds)

        # Write t_ci_bounds back into COMET's results PKL so that
        # specification_curve can find the CI column
        import pickle
        pkl_path = MV_DIR / "results" / "multiverse_results.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                comet_store = pickle.load(f)
            for i, row in df.iterrows():
                key = f"universe_{int(row['universe_id'])}"
                if key in comet_store:
                    comet_store[key]["t_ci_bounds"] = row["t_ci_bounds"]
            with open(pkl_path, "wb") as f:
                pickle.dump(comet_store, f)
            print(f"  Updated PKL with CI column ({len(comet_store)} universes)")
        else:
            print("  WARNING: PKL not found — CI may not work")

        n_with_ci = (df["t_ci_bounds"].apply(
            lambda x: not (x[0] != x[0])
        )).sum()
        print(f"  Universes with CI bounds: {n_with_ci} / {len(df)}")

        # Specification curve
        mverse.specification_curve(
            measure      = "group_t_mean",
            baseline     = 0,
            p_value      = "group_test_p",
            ci           = "t_ci_bounds",
            smooth_ci    = True,
            cmap         = "Set3",
            figsize      = (10, 9),
            fontsize     = 10,
            height_ratio = (1, 1),
            line_pad     = 0.1,
            ftype        = "png",
            dpi          = 150,
        )

        # Multiverse plot — density distribution across universes
        mverse.multiverse_plot(
            measure    = "group_t_mean",
            n_bins     = 10,
            sig_col    = "group_test_p",
            name_map   = name_map,
            baseline   = 0,
            figsize    = (7, 10),
            ftype      = "png",
            dpi        = 150,
        )

        # Decision structure DAG
        import matplotlib.pyplot as plt
        mverse.visualize(figsize=(16, 8), text_size=10, node_size=4000)
        plt.savefig(
            comet_results_dir / "multiverse_dag.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Copy all PNG figures from COMET results dir to outputs/
        # Uses glob to handle COMET's auto-numbering (e.g. specification_curve2.png)
        import shutil

        copy_map = {
            "specification":  "specification_curve.png",
            "multiverse_plot": "multiverse_plot.png",
            "multiverse_dag": "multiverse_dag.png",
            "multiverse":     "multiverse_overview.png",
        }

        for prefix, dst_name in copy_map.items():
            candidates = sorted(comet_results_dir.glob(f"{prefix}*.png"))
            if candidates:
                src_path = max(candidates, key=lambda p: p.stat().st_mtime)
                dst_path = DIR_MULTIVERSE_OUTPUTS / dst_name
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path.name} -> {dst_path.name}")
            else:
                print(f"Not found (skipping): {prefix}*.png")

elif MODE == "all":
    mverse.create(analysis_template, forking_paths)
    mverse.summary()
    mverse.run(parallel=1)
    df = mverse.get_results(as_df=True)
    if df is not None and len(df) > 0:
        out_tsv = DIR_MULTIVERSE_OUTPUTS / "multiverse_outcomes.tsv"
        df.to_csv(out_tsv, sep="\t", index=False)
        print(f"Results saved -> {out_tsv}")
        print(f"Universes: {len(df)}  |  "
              f"t-mean range: {df['group_t_mean'].min():.2f} "
              f"to {df['group_t_mean'].max():.2f}")

        comet_results_dir = MV_DIR / "results"

        # Human-readable labels for decision axes
        name_map = {
            "use_asr":       "Artifact subspace reconstruction",
            "brain_thresh":  "ICLabel brain threshold",
            "highpass_hz":   "High-pass filter (Hz)",
            "lowpass_hz":    "Low-pass filter (Hz)",
            "baseline_type": "ERSP baseline",
        }

        # Per-subject CI bounds for specification curve
        import ast as _ast
        import scipy.stats as _stats

        def _compute_ci_bounds(cell, confidence=0.95):
            try:
                rows = _ast.literal_eval(str(cell))
                t_vals = [
                    float(r["t_stat"])
                    for r in rows
                    if isinstance(r.get("t_stat"), (int, float))
                    and not (r["t_stat"] != r["t_stat"])
                ]
                if len(t_vals) >= 2:
                    mean  = float(_stats.tmean(t_vals))
                    se    = float(_stats.tsem(t_vals))
                    df_n  = len(t_vals) - 1
                    h     = se * _stats.t.ppf((1 + confidence) / 2, df_n)
                    return (mean - h, mean + h)
                elif len(t_vals) == 1:
                    return (t_vals[0], t_vals[0])
                else:
                    return (float("nan"), float("nan"))
            except Exception:
                return (float("nan"), float("nan"))

        df["t_ci_bounds"] = df["per_subject"].apply(_compute_ci_bounds)

        # Write t_ci_bounds back into COMET's results PKL so that
        # specification_curve can find the CI column
        import pickle
        pkl_path = MV_DIR / "results" / "multiverse_results.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                comet_store = pickle.load(f)
            for i, row in df.iterrows():
                key = f"universe_{int(row['universe_id'])}"
                if key in comet_store:
                    comet_store[key]["t_ci_bounds"] = row["t_ci_bounds"]
            with open(pkl_path, "wb") as f:
                pickle.dump(comet_store, f)
            print(f"  Updated PKL with CI column ({len(comet_store)} universes)")
        else:
            print("  WARNING: PKL not found — CI may not work")

        n_with_ci = (df["t_ci_bounds"].apply(
            lambda x: not (x[0] != x[0])
        )).sum()
        print(f"  Universes with CI bounds: {n_with_ci} / {len(df)}")

        # Specification curve
        mverse.specification_curve(
            measure      = "group_t_mean",
            baseline     = 0,
            p_value      = "group_test_p",
            ci           = "t_ci_bounds",
            smooth_ci    = True,
            cmap         = "Set3",
            figsize      = (10, 9),
            fontsize     = 10,
            height_ratio = (1, 1),
            line_pad     = 0.1,
            ftype        = "png",
            dpi          = 150,
        )

        # Multiverse plot — density distribution across universes
        mverse.multiverse_plot(
            measure    = "group_t_mean",
            n_bins     = 10,
            sig_col    = "group_test_p",
            name_map   = name_map,
            baseline   = 0,
            figsize    = (7, 10),
            ftype      = "png",
            dpi        = 150,
        )

        # Decision structure DAG
        import matplotlib.pyplot as plt
        mverse.visualize(figsize=(16, 8), text_size=10, node_size=4000)
        plt.savefig(
            comet_results_dir / "multiverse_dag.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Copy all PNG figures from COMET results dir to outputs/
        # Uses glob to handle COMET's auto-numbering (e.g. specification_curve2.png)
        import shutil

        copy_map = {
            "specification":  "specification_curve.png",
            "multiverse_plot": "multiverse_plot.png",
            "multiverse_dag": "multiverse_dag.png",
            "multiverse":     "multiverse_overview.png",
        }

        for prefix, dst_name in copy_map.items():
            candidates = sorted(comet_results_dir.glob(f"{prefix}*.png"))
            if candidates:
                src_path = max(candidates, key=lambda p: p.stat().st_mtime)
                dst_path = DIR_MULTIVERSE_OUTPUTS / dst_name
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path.name} -> {dst_path.name}")
            else:
                print(f"Not found (skipping): {prefix}*.png")

else:
    print(f"Unknown MODE '{MODE}'. Choose: create / run / universe / results / all")
