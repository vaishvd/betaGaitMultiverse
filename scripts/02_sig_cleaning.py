import mne
import numpy as np
import matplotlib.pyplot as plt
from src.paths import get_dataset_dirs

DATASET = "splitbelt"
SUBJECTS = ["S18"]

dirs = get_dataset_dirs(DATASET)

INPUT_DIR  = dirs["seg"]
OUTPUT_DIR = dirs["sigclean"]
MONTAGE_DIR = dirs["montage"]

TARGET_SFREQ = 512  
L_FREQ       = 1.0  
LINE_FREQS   = [60]  

SUBJECTS = ["S18"]

for subject in SUBJECTS:
    print(f"\nProcessing {subject}")

    raw = mne.io.read_raw_fif(
        INPUT_DIR / f"sub-{subject}_preadapt_raw.fif", preload=True
    )

    # Pick EEG channels only
    raw.pick("eeg")

    # Strip "1-" prefix added by BioSemi BDF import if present
    rename = {ch: ch.replace("1-", "", 1) for ch in raw.ch_names}
    raw.rename_channels(rename)

    # Montage
    montage = mne.channels.make_standard_montage("biosemi128")
    raw.set_montage(montage, on_missing="warn")
    print(f"  Channels: {len(raw.ch_names)} | sfreq: {raw.info['sfreq']:.0f} Hz")
    # Visualize the montage
    fig = montage.plot(kind='topomap', show_names=True)

    # Downsample
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ)
        print(f"  Downsampled → {TARGET_SFREQ} Hz")

    # High-pass filter
    raw.filter(l_freq=L_FREQ, h_freq=None, method="fir", fir_window="hamming")
    print(f"  High-pass filtered at {L_FREQ} Hz")

    # Notch filter
    raw.notch_filter(freqs=LINE_FREQS)
    print(f"  Notch filtered at {LINE_FREQS} Hz")

    # PSD check 
    fig = raw.compute_psd(fmax=80).plot(show=False)
    fig.savefig(OUTPUT_DIR / f"sub-{subject}_psd_sigclean.png", dpi=100)
    import matplotlib.pyplot as plt
    plt.close(fig)

    out = OUTPUT_DIR / f"sub-{subject}_clean_raw.fif"
    raw.save(out, overwrite=True)
    fig.savefig(MONTAGE_DIR / f"sub-{subject}_montage_layout.png", dpi=300)
    print(f"  Saved → {out.name}")

