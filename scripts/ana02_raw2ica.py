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

import mne
import numpy as np
from autoreject import AutoReject
from src.paths import get_dataset_dirs
from src.preprocessing import save_epochs, drop_invalid_eeg_channels
from src.ica_utils import run_ica, save_ica_component_plots

DATASET  = "stepup"
SUBJECTS = ["S1"]
TASK     = "CS"

TARGET_SFREQ       = 512
L_FREQ             = 1.0
LINE_FREQ          = 50
BAD_CHAN_THRESHOLD = 3.0
EPOCH_DUR          = 2.0
N_COMPONENTS       = 0.99
RANDOM_STATE       = 42

dirs     = get_dataset_dirs(DATASET)
RAW_DIR  = dirs["raw"]
PREP_DIR = dirs["prep"]

for subject in SUBJECTS:

    print(f"\n{'='*60}")
    print(f"ICA FIT: sub-{subject}")
    print(f"{'='*60}")

    # ── Load raw EEG ───────────────────────────────────────────────────────

    vhdr_file = (
        RAW_DIR / f"sub-{subject}" / "eeg"
        / f"sub-{subject}_task-{TASK}.vhdr"
    )

    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    print(f"  Loaded: {len(raw.ch_names)} channels | {raw.info['sfreq']:.0f} Hz")

    # ── Montage ────────────────────────────────────────────────────────────

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    # ── Resample ───────────────────────────────────────────────────────────

    if raw.info["sfreq"] > TARGET_SFREQ:
        raw.resample(TARGET_SFREQ)
        print(f"  Resampled → {TARGET_SFREQ} Hz")

    # ── Filter ─────────────────────────────────────────────────────────────

    raw.filter(l_freq=L_FREQ, h_freq=None, fir_design="firwin")
    raw.notch_filter(freqs=LINE_FREQ)
    print(f"  Filtered: high-pass {L_FREQ} Hz, notch {LINE_FREQ} Hz")

    # ── Bad channel detection ──────────────────────────────────────────────

    data = raw.get_data()
    var  = np.var(data, axis=1)
    z    = (var - var.mean()) / var.std()
    bads = [raw.ch_names[i] for i in np.where(np.abs(z) > BAD_CHAN_THRESHOLD)[0]]
    raw.info["bads"] = bads
    print(f"  Bad channels: {bads if bads else 'none'}")

    # ── Average reference — applied directly to data (not as projection) ──

    raw.set_eeg_reference("average", projection=False)
    print("  Re-referenced: average (applied to data)")

    # ── QC: PSD ────────────────────────────────────────────────────────────

    fig = raw.compute_psd(fmax=80, reject_by_annotation=False).plot(show=False)
    fig.savefig(PREP_DIR / f"sub-{subject}_psd.png")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  PSD saved")

    # ── Save preprocessed continuous raw ──────────────────────────────────

    clean_raw_out = PREP_DIR / f"sub-{subject}_clean_raw.fif"
    raw.save(clean_raw_out, overwrite=True)

    nan_frac = np.mean(np.isnan(raw.get_data()))
    print(f"  Saved clean_raw.fif  (NaN: {nan_frac*100:.1f}%)")

    if nan_frac > 0:
        print("  ERROR: NaN in saved raw — check preprocessing steps above.")
        continue

    # ── Epochs for ICA fitting ─────────────────────────────────────────────

    events = mne.make_fixed_length_events(raw, duration=EPOCH_DUR)
    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=EPOCH_DUR,
        baseline=None, preload=True,
        reject_by_annotation=False,
        verbose=False,
    )

    epochs.set_montage("standard_1020", on_missing="ignore")
    epochs = epochs.pick_types(eeg=True, eog=False, misc=False)
    epochs = drop_invalid_eeg_channels(epochs)
    print(f"  Epochs: {len(epochs)} × {len(epochs.ch_names)} ch")

    # ── AutoReject ─────────────────────────────────────────────────────────

    ar = AutoReject(n_interpolate=[1, 2, 4], random_state=RANDOM_STATE, n_jobs=1)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)

    n_bad = int(reject_log.bad_epochs.sum())
    reject_pct = 100 * n_bad / len(epochs)
    print(f"  AutoReject: {n_bad}/{len(epochs)} epochs rejected ({reject_pct:.1f}%)")

    if reject_pct > 30:
        print("  WARNING: >30% epochs rejected — check recording quality")
    if len(epochs_clean) < 20:
        raise RuntimeError(
            f"Too few clean epochs ({len(epochs_clean)}) — stopping pipeline."
        )

    save_epochs(epochs_clean, PREP_DIR, subject)

    # ── Fit ICA (no component exclusion — that is done in ana03) ──────────

    ica = run_ica(epochs_clean, n_components=N_COMPONENTS, random_state=RANDOM_STATE)

    ica_out = PREP_DIR / f"sub-{subject}_ica.fif"
    ica.save(ica_out, overwrite=True)
    print(f"  ICA saved: {ica.n_components_} components, none excluded")

    save_ica_component_plots(ica, PREP_DIR, subject)

print("\nICA FIT COMPLETE")
