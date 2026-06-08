"""
ana02_raw2ica.py
================
Preprocess continuous EEG and fit ICA decomposition.
IC rejection is handled separately in ana03_ica2clean.py.

Input
-----
d00_raw/  sub-{sub}/eeg/sub-{sub}_task-{TASK}.vhdr

Output
------
d02_prep/
    sub-{sub}_clean_raw.fif          preprocessed continuous raw
    sub-{sub}_preica_clean_epo.fif   AutoReject-cleaned epochs for ICA fit
    sub-{sub}_ica.fif                fitted ICA (no components excluded)
    sub-{sub}_ica_topos_*.png        component topography plots
    sub-{sub}_psd.png                PSD quality check
"""

import matplotlib.pyplot as plt
import mne
import numpy as np
from autoreject import AutoReject

from src.paths import get_dataset_dirs
from src.preprocessing import drop_invalid_channels, save_epochs, drop_invalid_eeg_channels
from src.ica_utils import run_ica, save_ica_component_plots

DATASET  = "stepup"
SUBJECTS = ["S1"]
TASK     = "CS"

TARGET_SFREQ = 250
L_FREQ = 1.0
LINE_FREQ = 50
BAD_CHAN_THRESHOLD = 3.0
EPOCH_DUR = 2.0
N_COMPONENTS = 0.99
RANDOM_STATE = 42

dirs = get_dataset_dirs(DATASET)
RAW_DIR  = dirs["raw"]
PREP_DIR = dirs["prep"]

for subject in SUBJECTS:

    print(f"\nProcessing sub-{subject}")

    vhdr_file = (
        RAW_DIR / f"sub-{subject}" / "eeg"
        / f"sub-{subject}_task-{TASK}.vhdr"
    )

    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
    raw.pick_types(eeg=True)

    raw = drop_invalid_channels(raw)

    # Montage

    raw.pick("eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    raw.plot_sensors(show_names=True)

    # Resample

    if raw.info["sfreq"] > TARGET_SFREQ:
        raw.resample(TARGET_SFREQ)

    # Filtering

    raw.filter(l_freq=L_FREQ, h_freq=60, fir_design="firwin")
    raw.notch_filter(freqs=LINE_FREQ)

    # BAD CHANNEL DETECTION 

    data = raw.get_data()

    ptp = np.ptp(data, axis=1)
    z = (ptp - np.mean(ptp)) / np.std(ptp)

    bad_idx = np.where(np.abs(z) > BAD_CHAN_THRESHOLD)[0]
    bads = [raw.ch_names[i] for i in bad_idx]

    print(f"  Bad channels detected: {bads}")


    # INTERPOLATE 

    raw.info["bads"] = bads
    if len(bads) > 0:
        raw.interpolate_bads(reset_bads=True)


    # Average reference

    raw.set_eeg_reference("average", projection=False)

    # Save clean raw

    clean_raw_out = PREP_DIR / f"sub-{subject}_clean_raw.fif"
    raw.save(clean_raw_out, overwrite=True)

    print(f"  Saved clean raw → {clean_raw_out.name}")

    # ICA EPOCHS

    events = mne.make_fixed_length_events(raw, duration=EPOCH_DUR)

    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=EPOCH_DUR,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )

    epochs = epochs.pick_types(eeg=True)

    epochs = drop_invalid_eeg_channels(epochs)

    print(f"  Epochs: {len(epochs)} × {len(epochs.ch_names)} ch")

    # AutoReject

    ar = AutoReject(
        n_interpolate=[1, 2, 4],
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    ar.fit(epochs)

    epochs_clean, reject_log = ar.transform(epochs, return_log=True)

    print(f"  Clean epochs: {len(epochs_clean)}")

    if len(epochs_clean) < 20:
        raise RuntimeError("Too few clean epochs after AutoReject")

    save_epochs(epochs_clean, PREP_DIR, subject)

    # ICA
    ica = run_ica(
        epochs_clean,
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE,
    )

    ica.save(PREP_DIR / f"sub-{subject}_ica.fif", overwrite=True)

    save_ica_component_plots(ica, PREP_DIR, subject)

print("\nICA FIT COMPLETE")