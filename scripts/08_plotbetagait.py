import numpy as np
import mne

from src.config import DIR_ERSP, DIR_ICA, DIR_PLOTS
from src.plot import plot_ersp_beta

SUBJECTS  = ["S18"]
CHANNELS   = ["C31","A30", "C32"]   # channels of interest — change to any labels in your montage
FREQS     = np.arange(13, 31)

for sub in SUBJECTS:
    print(f"\nProcessing {sub}")

    ersp = np.load(DIR_ERSP / f"sub-{sub}_ersp_beta.npy")
    # shape: (n_cycles, n_channels, n_freqs, n_timepoints)

    # Get channel index from the raw file's channel list
    raw    = mne.io.read_raw_fif(DIR_ICA / f"sub-{sub}_desc-clean_raw.fif", preload=False)
    ch_idx = [raw.ch_names.index(ch) for ch in CHANNELS]

    out_file = DIR_PLOTS / f"sub-{sub}_beta_ersp.png"
    plot_ersp_beta(ersp, FREQS, ch_idx, CHANNELS, sub, out_file)
    print(f"  Saved plot → {out_file.name}")