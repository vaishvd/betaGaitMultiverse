"""
ana05_gaitcycles2tfr.py
=======================
Compute gait-cycle ERSP using Morlet TFR with standing baseline.

Preprocessing before TFR:
  1. Apply the same ICA cleaning to the standing baseline that was applied
     to the gait data — so both are in the same signal space.
  2. Apply a Current Source Density (CSD / Surface Laplacian) transform to
     both gait epochs and standing baseline.  CSD weights each electrode
     against its neighbours with distance-decaying coefficients (high at
     the electrode, decaying outward), emphasising local cortical sources
     and removing spatially broad volume-conducted activity.

Input
-----
d04_gaitepochs/  sub-{sub}_gait_epo.fif
d03_clean/       sub-{sub}_ica-clean.fif   ICA with exclusions list
d00_raw/         sub-{sub}_task-STAND.vhdr

Output
------
d05_ersp/  sub-{sub}_ersp_beta.npy   (n_ch × n_freqs × n_time)
"""

import numpy as np
import mne
import warnings

from src.paths import get_dataset_dirs
from src.preprocessing import filter_raw, rereference_raw


DATASET  = "stepup"
SUBJECTS = ["S1"]

FREQS    = np.arange(13, 31, dtype=float)   # beta band
N_CYCLES = FREQS / 2.0

EDGE_CROP    = 0.05   # fraction of cycle to discard at each edge (wavelet artefact)
TARGET_SFREQ = 512
L_FREQ       = 1.0
LINE_FREQS   = [50]

dirs = get_dataset_dirs(DATASET)

GAITEPOCH_DIR = dirs["gaitepochs"]
CLEAN_DIR     = dirs["clean"]
ERSP_DIR      = dirs["ersp"]
RAW_DIR       = dirs["raw"]


for subject in SUBJECTS:

    print(f"\n{'='*60}")
    print(f"ERSP: sub-{subject}")
    print(f"{'='*60}")

    #  Load gait epochs

    epochs = mne.read_epochs(
        GAITEPOCH_DIR / f"sub-{subject}_gait_epo.fif",
        preload=True, verbose=False,
    )
    sfreq = epochs.info["sfreq"]
    print(f"  Gait epochs : {len(epochs)} × {len(epochs.ch_names)} ch")
    print(f"  sfreq       : {sfreq:.0f} Hz  (= N_POINTS, normalised time axis)")

    #  Load ICA and check alignment of gait cycles with EEG recording

    ica = mne.preprocessing.read_ica(
        CLEAN_DIR / f"sub-{subject}_ica-clean.fif",
        verbose=False,
    )
    print(f"  ICA loaded  : {ica.n_components_} components, "
          f"{len(ica.exclude)} excluded")

    # Load and preprocess standing baseline raw

    stand_file = (
        RAW_DIR / f"sub-{subject}" / "eeg"
        / f"sub-{subject}_task-STAND.vhdr"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_stand = mne.io.read_raw_brainvision(
            stand_file, preload=True, verbose=False,
        )

    raw_stand.pick_types(eeg=True, verbose=False)

    # Same preprocessing as gait data
    raw_stand.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        on_missing="ignore",
    )
    filter_raw(raw_stand, TARGET_SFREQ, L_FREQ, LINE_FREQS)
    rereference_raw(raw_stand, ref_type="average", plot=False)

    # Match channels to gait epochs (same set as used for ICA and TFR)
    raw_stand.pick(epochs.ch_names)

    # Apply the same ICA cleaning as gait data — ensures both are in
    # the same signal space before computing the power ratio
    raw_stand = ica.apply(raw_stand.copy(), verbose=False)
    print(f"  Standing raw: {raw_stand.n_times} samples, "
          f"{raw_stand.info['sfreq']:.0f} Hz, "
          f"ICA-cleaned with {len(ica.exclude)} components removed")

    # CSD (Surface Laplacian)
    # Weights each electrode against its neighbours with Laplacian coefficients
    # that decay outward from centre → localises cortical sources,
    # removes volume-conducted activity common to both conditions.

    print("\n  Applying CSD (Surface Laplacian) ...")
    epochs_csd   = mne.preprocessing.compute_current_source_density(
        epochs, verbose=False,
    )
    raw_stand_csd = mne.preprocessing.compute_current_source_density(
        raw_stand, verbose=False,
    )
    print(f"  CSD applied: {len(epochs_csd.ch_names)} channels")

    # Standing baseline TFR 

    stand_data = raw_stand_csd.get_data()

    baseline_tfr = mne.time_frequency.tfr_array_morlet(
        stand_data[np.newaxis],
        sfreq=raw_stand_csd.info["sfreq"],
        freqs=FREQS,
        n_cycles=N_CYCLES,
        output="power",
        zero_mean=True,
        verbose=False,
    )[0]                                        # (n_ch, n_freqs, n_time)

    baseline = baseline_tfr.mean(axis=-1, keepdims=True)   # (n_ch, n_freqs, 1)
    print(f"  Baseline shape : {baseline.shape}")

    #  Gait TFR per cycle 

    gait_data  = epochs_csd.get_data()          # (n_epochs, n_ch, N_POINTS)
    tfr_cycles = []

    for cycle in gait_data:
        power = mne.time_frequency.tfr_array_morlet(
            cycle[np.newaxis],
            sfreq=sfreq,
            freqs=FREQS,
            n_cycles=N_CYCLES,
            output="power",
            zero_mean=True,
            verbose=False,
        )[0]                                    # (n_ch, n_freqs, n_time)

        crop = int(EDGE_CROP * power.shape[-1])
        if crop > 0:
            power = power[..., crop:-crop]

        tfr_cycles.append(power)

    tfr_cycles = np.stack(tfr_cycles)           # (n_epochs, n_ch, n_freqs, n_time)
    print(f"  TFR shape  : {tfr_cycles.shape}")

    #  ERSP normalisation 

    ersp     = 10 * np.log10(tfr_cycles / baseline)
    ersp_avg = ersp.mean(axis=0)               # (n_ch, n_freqs, n_time)

    print(f"  ERSP shape : {ersp_avg.shape}")
    print(f"  ERSP range : {ersp_avg.min():.2f} to {ersp_avg.max():.2f} dB")


    out = ERSP_DIR / f"sub-{subject}_ersp_beta.npy"
    np.save(out, ersp_avg)
    print(f"  Saved → {out.name}")

print("\nERSP COMPUTATION COMPLETE")
