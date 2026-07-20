"""
betaGaitMultiverse — visualize multiverse results.

Produces a specification curve, multiverse density plot, and
visualizes the multiverse graph. Requires mulana02_run_multiverse.py
to have completed successfully.
"""

from comet.multiverse import Multiverse
from src.config import MULTIVERSE_NAME, DIR_MULTIVERSE_COMET

# path= must match mulana01's -- see src.config.DIR_MULTIVERSE_COMET.
mverse = Multiverse(name=MULTIVERSE_NAME, path=str(DIR_MULTIVERSE_COMET))

# --- Specification curve ---
name_map = {
    "t_stats":        "Subject beta contrast\n(double stance - swing, dB)",
    "highpass_hz":    "High-pass (Hz)",
    "asr_mode":       "ASR",
    "iclabel_rule":   "IC selection",
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
