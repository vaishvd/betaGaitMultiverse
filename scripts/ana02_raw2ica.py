import mne
import numpy as np
from autoreject import AutoReject
from src.paths import get_dataset_dirs
from pathlib import Path
from src.preprocessing import save_epochs
from src.ica_utils import (
    run_ica,
    label_and_mark_ica,
    save_ica_component_plots,
)

DATASET  = "stepup"
SUBJECTS = ["S1"]
TASK = "CS"
DATATYPE = "eeg"

# Filtering
TARGET_SFREQ = 512
L_FREQ       = 1.0
LINE_FREQ   = 50

# Bad channels
BAD_CHAN_THRESHOLD = 3.0

# Epoching
EPOCH_DUR = 2.0

# ICA
N_COMPONENTS = 0.99
RANDOM_STATE = 42

# ICLabel
BRAIN_THRESH = 0.7

dirs = get_dataset_dirs(DATASET)

RAW_DIR  = dirs["raw"]
PREP_DIR = dirs["prep"]

for subject in SUBJECTS:

    print(f"\n{'='*60}")
    print(f"sub-{subject}")
    print(f"{'='*60}")

    # Load raw EEG
    vhdr_file = (
        RAW_DIR
        / f"sub-{subject}"
        / "eeg"
        / f"sub-{subject}_task-{TASK}.vhdr"
    )

    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

    raw.pick_types(eeg=True)

    print(f"Loaded: {len(raw.ch_names)} channels | {raw.info['sfreq']} Hz")

    # Montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    print(f"  Channels: {len(raw.ch_names)} | sfreq: {raw.info['sfreq']:.0f} Hz")
    # Visualize the montage
    fig = montage.plot(kind='topomap', show_names=True)

    # Downsample
    if raw.info["sfreq"] > TARGET_SFREQ:
        raw.resample(TARGET_SFREQ)
        print(f"Downsampled → {TARGET_SFREQ} Hz")

    # High-pass filter
    raw.filter(l_freq=L_FREQ, h_freq=None, fir_design="firwin")
    print(f"  High-pass filtered at {L_FREQ} Hz")

    # Notch filter
    raw.notch_filter(freqs=LINE_FREQ)
    print(f"  Notch filtered at {LINE_FREQ} Hz")

    # Bad channel detection
    data = raw.get_data()
    var = np.var(data, axis=1)
    z = (var - var.mean()) / var.std()

    bads = [raw.ch_names[i] for i in np.where(np.abs(z) > 3.0)[0]]
    raw.info["bads"] = bads

    print(f"Bads detected: {bads}")
    # Average re-reference 
    raw.set_eeg_reference("average")
    print("  Re-referenced to average")

    psd = raw.compute_psd(fmax=80, reject_by_annotation=False)
    fig = psd.plot(show=False)
    fig.savefig(PREP_DIR / f"sub-{subject}_psd.png")
    print("PSD saved")

# Epoching for AutoReject

    events = mne.make_fixed_length_events(raw, duration=EPOCH_DUR)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=EPOCH_DUR,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
    )

    print(f"Epochs created: {len(epochs)}")

    ar = AutoReject(
        n_interpolate=[1, 2, 4],
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    epochs_clean = ar.fit_transform(epochs)

    print(f"AutoReject done: {len(epochs_clean)} kept")

    save_epochs(epochs_clean, PREP_DIR, subject)

# ICA
    ica = run_ica(
        epochs_clean,
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE,
    )

    label_and_mark_ica(
        ica,
        epochs_clean,
        brain_thresh=BRAIN_THRESH,
    )

    ica_out = PREP_DIR / f"sub-{subject}_ica.fif"
    ica.save(ica_out, overwrite=True)

    print("ICA saved")

    # ICA QC plots
    save_ica_component_plots(
        ica,
        PREP_DIR,
        subject,
    )

print("\nDone.")