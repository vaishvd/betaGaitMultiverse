import numpy as np
import mne
from src.config import DIR_GAIT, DIR_TFR, DIR_ICA
from src.tfr import compute_tfr

SUBJECTS = ["S18"]

FREQS = np.arange(13, 31)  # beta band: 13–30 Hz

for sub in SUBJECTS:
    print(f"\nProcessing {sub}")

    data = np.load(DIR_GAIT / f"sub-{sub}_gait_cycles.npy")
    # shape: (n_cycles, n_channels, n_timepoints)

    raw   = mne.io.read_raw_fif(DIR_ICA / f"sub-{sub}_desc-clean_raw.fif", preload=False)
    sfreq = raw.info["sfreq"]

    power = compute_tfr(data, sfreq, FREQS)
    # shape: (n_cycles, n_channels, n_freqs, n_timepoints)

    np.save(DIR_TFR / f"sub-{sub}_tfr_beta.npy", power)
    print(f"  TFR shape {power.shape} → saved sub-{sub}_tfr_beta.npy")