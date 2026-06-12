"""
Publication-quality specification curve for the betaGaitMultiverse.

Reads results/multiverse/multiverse_outcomes.tsv and produces a
two-panel figure following Simonsohn et al. 2020 (PNAS):
  - Top panel: sorted group_t_mean with per-subject t-stat range
  - Bottom panel: decision grid showing active option per universe

Output: results/multiverse/outputs/specification_curve.png
"""

import ast
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import DIR_MULTIVERSE, DIR_MULTIVERSE_OUTPUTS
from src.multiverse import forking_paths

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
outcomes_path = DIR_MULTIVERSE_OUTPUTS / "multiverse_outcomes.tsv"
if not outcomes_path.exists():
    print(f"Results not found: {outcomes_path}")
    print("Run: python scripts/multiverse/run_multiverse.py --results")
    sys.exit(1)

df = pd.read_csv(outcomes_path, sep="\t")
print(f"Loaded {len(df)} universes")
print(f"Columns: {list(df.columns)}")

# Parse the decisions dict string (stored by COMET as repr of dict)
df["_dec"] = df["decisions"].apply(ast.literal_eval)

decision_keys = list(forking_paths.keys())
print(f"Decision keys: {decision_keys}")

# ---------------------------------------------------------------------------
# Load per-subject t-statistics from COMET's combined results PKL
# ---------------------------------------------------------------------------
combined_pkl = DIR_MULTIVERSE_OUTPUTS / "multiverse_results.pkl"
per_subject_t = []

if combined_pkl.exists():
    with open(combined_pkl, "rb") as f:
        combined = pickle.load(f)
    for i in range(1, len(df) + 1):
        key = f"universe_{i}"
        res = combined.get(key, {})
        t_vals = [
            r["t_stat"] for r in res.get("per_subject", [])
            if not np.isnan(r.get("t_stat", float("nan")))
        ]
        per_subject_t.append(t_vals)
    print(f"Per-subject data loaded from combined PKL: {sum(1 for v in per_subject_t if v)} / {len(df)} universes")
else:
    per_subject_t = [[] for _ in range(len(df))]
    print("WARNING: combined PKL not found — subject-range shading disabled")

# ---------------------------------------------------------------------------
# Sort universes by group_t_mean
# ---------------------------------------------------------------------------
df["_orig_idx"] = range(len(df))
df_sorted = df.sort_values("group_t_mean").reset_index(drop=True)
sorted_order = df_sorted["_orig_idx"].values

t_means = df_sorted["group_t_mean"].values
t_stds  = df_sorted["group_t_std"].values

t_per_sub_sorted = [per_subject_t[i] for i in sorted_order]
t_upper = np.array([
    np.max(v) if v else t_means[k] for k, v in enumerate(t_per_sub_sorted)
])
t_lower = np.array([
    np.min(v) if v else t_means[k] for k, v in enumerate(t_per_sub_sorted)
])

N = len(df_sorted)
xs = np.arange(N)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
n_dec = len(decision_keys)
fig = plt.figure(figsize=(12, 5 + n_dec * 0.55))
gs  = gridspec.GridSpec(
    2, 1,
    height_ratios=[3, n_dec],
    hspace=0.08,
    left=0.18, right=0.96, top=0.93, bottom=0.06,
)

ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

fig.suptitle(
    "Specification curve -- stance vs swing beta t-statistic",
    fontsize=11, fontweight="bold", y=0.97,
)

# ---------------------------------------------------------------------------
# Top panel
# ---------------------------------------------------------------------------
ax_top.fill_between(xs, t_lower, t_upper,
                    color="gray", alpha=0.20)

ax_top.plot(xs, t_means, color="black", linewidth=0.8, alpha=0.6, zorder=2)

sig = df_sorted["group_test_p"].values < 0.05
colors = ["#1D9E75" if s else "#444441" for s in sig]
ax_top.scatter(xs, t_means, c=colors, s=28, zorder=3,
               linewidths=0.4, edgecolors="white")

ax_top.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)

ax_top.set_ylabel("group mean t-stat\n(stance vs swing)", fontsize=9)
ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
ax_top.tick_params(axis="y", labelsize=8)
ax_top.spines[["top", "right", "bottom"]].set_visible(False)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D9E75",
           markersize=7, label="p < 0.05"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#444441",
           markersize=7, label="p >= 0.05"),
    plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.25, label="subject range"),
]
ax_top.legend(handles=legend_elements, fontsize=7, loc="upper left",
              framealpha=0.8)

# ---------------------------------------------------------------------------
# Bottom panel — decision grid
# ---------------------------------------------------------------------------
PALETTES = {
    "use_asr":       {str(False): "#378ADD", str(True): "#E24B4A"},
    "brain_thresh":  {"0.7": "#1D9E75",      "0.9": "#0F6E56"},
    "highpass_hz":   {"0.1": "#BA7517",       "2.0": "#EF9F27"},
    "lowpass_hz":    {"40":  "#7F77DD",        "None": "#E24B4A"},
    "baseline_type": {"standing": "#D4537E", "walking_mean": "#F0997B"},
}

ax_bot.set_xlim(-0.5, N - 0.5)
ax_bot.set_ylim(-0.5, n_dec - 0.5)
ax_bot.axis("off")

# Extract dicts as a plain Python list — list comprehension below preserves
# original Python types (avoids pandas float64 coercion of int/None columns).
decs_list = df_sorted["_dec"].tolist()

for row_idx, dec_key in enumerate(decision_keys):
    vals_sorted = [d[dec_key] for d in decs_list]
    palette     = PALETTES.get(dec_key, {})

    for col_idx, val in enumerate(vals_sorted):
        color = palette.get(str(val), "#AAAAAA")
        ax_bot.scatter(col_idx, row_idx, c=color, s=22, marker="o",
                       linewidths=0, zorder=2)

    # Row key label (left of grid)
    ax_bot.text(-1.2, row_idx, dec_key,
                ha="right", va="center", fontsize=8, color="#444441",
                transform=ax_bot.transData)

    # Option value labels (stacked, offset slightly above/below key label)
    opts = [str(v) for v in forking_paths[dec_key]]
    for opt_idx, opt in enumerate(opts):
        sign = 1 if opt_idx == 0 else -1
        color = palette.get(opt, "#888888")
        ax_bot.text(-1.2, row_idx + sign * 0.26, opt,
                    ha="right", va="center", fontsize=6.5, color=color,
                    transform=ax_bot.transData)

# Horizontal row separators
for row_idx in range(n_dec):
    ax_bot.axhline(row_idx - 0.5, color="#DDDDDD", linewidth=0.5, zorder=0)

ax_bot.text(N / 2, -0.45,
            f"Universe (n={N}, sorted by group mean t-statistic)",
            ha="center", va="top", fontsize=8, color="#666666",
            transform=ax_bot.transData)

# Decision influence annotation
influence = {
    "use_asr":       1.193,
    "lowpass_hz":    1.121,
    "highpass_hz":   0.617,
    "brain_thresh":  0.337,
    "baseline_type": 0.004,
}
influence_str = "Decision influence (range of cell means):  " + \
    "  |  ".join(f"{k}={v:.2f}" for k, v in influence.items())
fig.text(
    0.18, 0.01, influence_str,
    fontsize=7, color="#666666", ha="left", va="bottom",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_png = DIR_MULTIVERSE_OUTPUTS / "specification_curve.png"
fig.savefig(out_png, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_png}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print(f"\n=== Specification curve summary ===")
print(f"Universes:          {N}")
print(f"t-mean range:       {t_means.min():.2f} to {t_means.max():.2f}")
print(f"Positive universes: {(t_means > 0).sum()} / {N}")
print(f"Significant (p<0.05): {sig.sum()} / {N}")

print(f"\nDecision influence on group_t_mean (range of cell means):")
for dec_key in decision_keys:
    # Use Python-native types to avoid None->nan coercion
    vals = df_sorted["_dec"].apply(lambda d: str(d[dec_key])).tolist()
    cell_means = {}
    for val in sorted(set(vals)):
        mask = np.array([v == val for v in vals])
        cell_means[val] = t_means[mask].mean()
    spread = max(cell_means.values()) - min(cell_means.values())
    print(f"  {dec_key:>20}: range={spread:.3f}")
    for val, mean in cell_means.items():
        print(f"    {val:>12}: mean t = {mean:.3f}")
