"""
betaGaitMultiverse — visualize multiverse results.

Produces a specification curve, multiverse density plot, and
visualizes the multiverse graph. Requires mulana02_run_multiverse.py
to have completed successfully.
"""

from comet.multiverse import Multiverse
from src.config import MULTIVERSE_NAME

mverse = Multiverse(name=MULTIVERSE_NAME)

# --- Specification curve ---
name_map = {
    "t_stats":        "Subject t-statistics\n(double stance vs swing\nbeta power)",
    "use_asr":        "ASR\ndenoising",
    "use_gedai":      "GEDAI\ndenoising",
    "baseline_type":  "Baseline\ntype",
    "brain_thresh":   "ICLabel\nthreshold",
}

mverse.specification_curve(
    measure      = "t_stats",
    name_map     = name_map,
    p_value      = 0.05,
    ci           = 95,
    smooth_ci    = True,
    cmap         = "Set3",
    figsize      = (11, 8),
    fontsize     = 10,
    height_ratio = [1, 1],
    ftype        = "pdf",
)

# --- Multiverse density plot ---
name_map_density = {
    "t_stats":        "Subject t-statistics\n(double stance vs swing\nbeta power)",
    "use_asr":        "ASR\ndenoising",
    "use_gedai":      "GEDAI\ndenoising",
    "baseline_type":  "Baseline\ntype",
    "brain_thresh":   "ICLabel\nthreshold",
}

mverse.multiverse_plot(
    measure  = "group_t_mean",
    n_bins   = 4,
    name_map = name_map_density,
    baseline = 0,
)
