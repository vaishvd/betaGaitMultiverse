import numpy as np

from src.config import DIR_TFR, DIR_ERSP
from src.ersp import log_power, baseline_correct

SUBJECTS = ["S18"]

for sub in SUBJECTS:
    print(f"\nProcessing {sub}")

    power = np.load(DIR_TFR / f"sub-{sub}_tfr_beta.npy")
    # shape: (n_cycles, n_channels, n_freqs, n_timepoints)

    power_db = log_power(power)
    ersp     = baseline_correct(power_db)
    # Baseline: mean across ALL cycles and timepoints per channel/frequency.
    # This removes mean power level without assuming any cycle phase is neutral.

    np.save(DIR_ERSP / f"sub-{sub}_ersp_beta.npy", ersp)
    print(f"  ERSP shape {ersp.shape} → saved sub-{sub}_ersp_beta.npy")