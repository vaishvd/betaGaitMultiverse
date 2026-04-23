import numpy as np
import mne
from src.config import DIR_GAIT, DIR_TFR

SUBJECTS  = ["S18"]
FREQS     = np.arange(13, 31, dtype=float)   # beta band (Hz)
N_CYCLES  = FREQS / 2.0                      # frequency-dependent wavelet width
N_POINTS  = 512                              # gait cycle % resolution
EDGE_CROP = 0.05                             # fraction trimmed at each edge post-TFR


for sub in SUBJECTS:
    print(f"\n── {sub} ──")

    segments = np.load(DIR_GAIT / f"sub-{sub}_gait_segments.npy", allow_pickle=True)
    sfreq    = float(np.load(DIR_GAIT / f"sub-{sub}_gait_sfreq.npy"))

    tfr_cycles = []
    for data in segments:
        # Morlet at true sfreq — frequency resolution is physically meaningful
        power = mne.time_frequency.tfr_array_morlet(
            data[np.newaxis],
            sfreq=sfreq,
            freqs=FREQS,
            n_cycles=N_CYCLES,
            output="power",
            zero_mean=True,
        )[0]                                 # (channels, freqs, samples)

        # Crop wavelet edge artefacts in real-time samples before resampling
        crop = int(EDGE_CROP * power.shape[-1])
        if crop > 0:
            power = power[..., crop:-crop]

        # Resample time axis to N_POINTS (% gait cycle)
        x_old = np.linspace(0, 1, power.shape[-1])
        x_new = np.linspace(0, 1, N_POINTS)
        power = np.array([[np.interp(x_new, x_old, power[ch, fr])
                           for fr in range(power.shape[1])]
                          for ch in range(power.shape[0])])
        tfr_cycles.append(power)

    # (n_cycles, channels, freqs, n_points) — linear power, not yet baselined
    tfr_cycles = np.stack(tfr_cycles)
    print(f"  Shape : {tfr_cycles.shape}, any nan: {np.isnan(tfr_cycles).any()}")

    np.save(DIR_TFR / f"sub-{sub}_tfr_beta.npy", tfr_cycles)
    print(f"  Saved → sub-{sub}_tfr_beta.npy")