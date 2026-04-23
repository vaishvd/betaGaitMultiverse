import numpy as np
from src.config import DIR_TFR, DIR_ERSP

SUBJECTS       = ["S18"]
BASELINE   = (0, 10)     # % gait cycle — early stance, closest to rest - baceline window


for sub in SUBJECTS:
    print(f"\n {sub} ERSP computation")

    # (n_cycles, channels, freqs, n_points) — linear power
    tfr = np.load(DIR_TFR / f"sub-{sub}_tfr_beta.npy")

    n_points = tfr.shape[-1]
    b0 = int(BASELINE[0] / 100 * n_points)
    b1 = int(BASELINE[1] / 100 * n_points)

    # Baseline: mean linear power in early stance window, per cycle/channel/freq
    # Averaging in linear space before log conversion avoids log-space bias
    baseline = tfr[..., b0:b1].mean(axis=-1, keepdims=True)   # (n, ch, fr, 1)

    ersp     = 10 * np.log10(tfr / baseline)                  # (n, ch, fr, n_points)
    ersp_avg = ersp.mean(axis=0)                               # (ch, fr, n_points)

    print(f"  Baseline window : {BASELINE[0]}–{BASELINE[1]}% of cycle "
          f"({b1 - b0} samples)")
    print(f"  Shape  : {ersp_avg.shape}, any nan: {np.isnan(ersp_avg).any()}")
    print(f"  Range  : {ersp_avg.min():.2f} / {ersp_avg.max():.2f} dB")

    np.save(DIR_ERSP / f"sub-{sub}_ersp_beta.npy", ersp_avg)
    print(f"  Saved → sub-{sub}_ersp_beta.npy")