"""
Multiverse configuration for betaGaitMultiverse.

Defines MULTIVERSE_NAME and forking_paths — imported by
mulana01_run_multiverse.py and mulana02_plot_specification_curve.py.
"""

from src.config import MULTIVERSE_NAME

forking_paths = {
    "use_asr":       [False, True],
    "brain_thresh":  [0.7, 0.9],
    "highpass_hz":   [0.1, 2.0],
    "lowpass_hz":    [40, None],
    "baseline_type": ["standing", "walking_mean"],
}

__all__ = ["MULTIVERSE_NAME", "forking_paths"]
