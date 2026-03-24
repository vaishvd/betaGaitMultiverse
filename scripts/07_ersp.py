import numpy as np

from src.config import DIR_TFR, DIR_ERSP
from src.ersp import log_power, baseline_correct

SUBJECTS = ["S18"]
DIR_ERSP.mkdir(exist_ok=True)

# Baseline window as a fraction of the normalized gait cycle.
# (0.0, 0.1) = first 10% of the cycle (20 time points out of 200).
# Change to a rest-condition array for a more rigorous baseline.
BASELINE_PCT = (0.0, 0.1)

for sub in SUBJECTS:
    print(f"\nProcessing {sub}")

    power = np.load(DIR_TFR / f"sub-{sub}_tfr_beta.npy")
    # shape: (n_cycles, n_channels, n_freqs, n_timepoints)

    power_db = log_power(power)
    ersp     = baseline_correct(power_db, baseline_pct=BASELINE_PCT)
    # shape: (n_cycles, n_channels, n_freqs, n_timepoints)

    np.save(DIR_ERSP / f"sub-{sub}_ersp_beta.npy", ersp)
    print(f"  ERSP shape {ersp.shape} → saved sub-{sub}_ersp_beta.npy")