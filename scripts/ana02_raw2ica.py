"""
ana02_raw2ica.py
================
# Standing (STAND) and walking (CS) recordings are concatenated before
# preprocessing so that filtering, referencing, bad-channel interpolation,
# and ICA decomposition are applied identically to both conditions.
# ICA is fitted on epochs drawn from the full concatenated recording,
# giving more data for decomposition and ensuring the same component
# structure is used for both conditions.
# Condition boundaries are stored as annotations ("STAND", "CS") in the
# concatenated raw for downstream segment extraction.
# See: Makeig et al. 1996 J Neurosci; Delorme et al. 2012 Front Hum Neurosci

Input
-----
d00_raw/  sub-{sub}/eeg/sub-{sub}_task-STAND.vhdr
d00_raw/  sub-{sub}/eeg/sub-{sub}_task-CS.vhdr

Output
------
d02_prep/
    sub-{sub}_concat_raw.fif           preprocessed concatenated (STAND+CS) raw
    sub-{sub}_preica_clean_epo.fif     AutoReject-cleaned epochs for ICA fit
    sub-{sub}_ica.fif                  fitted ICA (no components excluded)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from autoreject import AutoReject

from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS
from src.preprocessing import drop_invalid_channels, save_epochs, drop_invalid_eeg_channels
from src.ica_utils import run_ica, save_ica_component_plots
from src.qc import log_qc

TASK_STAND = "STAND"
TASK_WALK  = "CS"

TARGET_SFREQ       = 250
L_FREQ             = 1.0
LINE_FREQ          = 50
BAD_CHAN_THRESHOLD = 3.0
EPOCH_DUR          = 2.0
N_COMPONENTS       = 0.99
RANDOM_STATE       = 42

dirs     = get_dataset_dirs(DATASET)
RAW_DIR  = dirs["raw"]
PREP_DIR = dirs["prep"]
QC_DIR   = dirs["qc"]

for subject in SUBJECTS:
    try:
        stand_path = RAW_DIR / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-STAND.vhdr"
        walk_path  = RAW_DIR / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-CS.vhdr"
        missing = [p for p in [stand_path, walk_path] if not p.exists()]
        if missing:
            for p in missing:
                print(f"  [SKIP] sub-{subject}: missing {p.name}")
            continue

        print(f"\nProcessing sub-{subject}")

        stand_vhdr = (
            RAW_DIR / f"sub-{subject}" / "eeg"
            / f"sub-{subject}_task-{TASK_STAND}.vhdr"
        )
        walk_vhdr = (
            RAW_DIR / f"sub-{subject}" / "eeg"
            / f"sub-{subject}_task-{TASK_WALK}.vhdr"
        )

        # Load STAND and CS recordings

        print("  Loading STAND recording ...")
        raw_stand = mne.io.read_raw_brainvision(stand_vhdr, preload=True, verbose=False)
        raw_stand.pick_types(eeg=True)
        raw_stand = drop_invalid_channels(raw_stand)
        raw_stand.pick("eeg")
        montage = mne.channels.make_standard_montage("standard_1005")
        raw_stand.set_montage(montage, on_missing="ignore")

        print("  Loading CS (walking) recording ...")
        raw_walk = mne.io.read_raw_brainvision(walk_vhdr, preload=True, verbose=False)
        raw_walk.pick_types(eeg=True)
        raw_walk = drop_invalid_channels(raw_walk)
        raw_walk.pick("eeg")
        raw_walk.set_montage(montage, on_missing="ignore")

        # Verify compatibility of STAND and CS recordings before concatenation

        if raw_stand.ch_names != raw_walk.ch_names:
            raise RuntimeError(
                f"sub-{subject}: STAND and CS have different channel names.\n"
                f"  STAND ({len(raw_stand.ch_names)} ch): {raw_stand.ch_names[:5]}...\n"
                f"  CS    ({len(raw_walk.ch_names)} ch): {raw_walk.ch_names[:5]}..."
            )
        if raw_stand.info["sfreq"] != raw_walk.info["sfreq"]:
            raise RuntimeError(
                f"sub-{subject}: STAND sfreq ({raw_stand.info['sfreq']}) != "
                f"CS sfreq ({raw_walk.info['sfreq']})"
            )

        print(f"  STAND: {raw_stand.n_times} samples  {raw_stand.times[-1]:.1f} s  "
              f"sfreq={raw_stand.info['sfreq']:.0f} Hz")
        print(f"  CS   : {raw_walk.n_times} samples  {raw_walk.times[-1]:.1f} s")

        #  Concatenate and add condition annotations

        # Compute segment boundaries before concatenation (in seconds at original sfreq)
        stand_dur  = raw_stand.times[-1]
        walk_dur   = raw_walk.times[-1]
        walk_onset = float(raw_stand.n_times) / raw_stand.info["sfreq"]

        raw_concat = mne.concatenate_raws([raw_stand, raw_walk], preload=True)

        # Annotate condition boundaries for downstream segment extraction
        raw_concat.annotations.append(onset=0.0,        duration=stand_dur, description="STAND")
        raw_concat.annotations.append(onset=walk_onset, duration=walk_dur,  description="CS")

        print(f"  Concatenated: {raw_concat.n_times} samples  "
              f"{raw_concat.times[-1]:.1f} s  sfreq={raw_concat.info['sfreq']:.0f} Hz")

        # Preprocess concatenated raw

        if raw_concat.info["sfreq"] > TARGET_SFREQ:
            raw_concat.resample(TARGET_SFREQ)
            print(f"  Resampled to {TARGET_SFREQ} Hz")

        raw_concat.filter(l_freq=L_FREQ, h_freq=60, fir_design="firwin")
        raw_concat.notch_filter(freqs=LINE_FREQ)

        data    = raw_concat.get_data()
        ptp     = np.ptp(data, axis=1)
        z       = (ptp - np.mean(ptp)) / np.std(ptp)
        bad_idx = np.where(np.abs(z) > BAD_CHAN_THRESHOLD)[0]
        bads    = [raw_concat.ch_names[i] for i in bad_idx]
        print(f"  Bad channels detected: {bads}")

        raw_concat.info["bads"] = bads
        if len(bads) > 0:
            raw_concat.interpolate_bads(reset_bads=True)

        raw_concat.set_eeg_reference("average", projection=False)

        # Save preprocessed concatenated raw

        concat_out = PREP_DIR / f"sub-{subject}_concat_raw.fif"
        raw_concat.save(concat_out, overwrite=True)
        print(f"  Saved concat raw -> {concat_out.name}")

        # ICA training epochs from full concatenated raw

        events = mne.make_fixed_length_events(raw_concat, duration=EPOCH_DUR)
        epochs = mne.Epochs(
            raw_concat, events,
            tmin=0, tmax=EPOCH_DUR,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            verbose=False,
        )
        epochs = epochs.pick_types(eeg=True)
        epochs = drop_invalid_eeg_channels(epochs)
        print(f"  Epochs: {len(epochs)} x {len(epochs.ch_names)} ch")

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

        # Fit ICA

        # Extended Infomax ICA (Lee et al. 1999, Neural Computation 11:417-441).
        # Adaptively estimates sub- and super-Gaussian sources, making it more
        # appropriate for EEG than standard FastICA which assumes super-Gaussian
        # sources only. Extended Infomax is the decomposition used to train
        # ICLabel (Pion-Tonachini et al. 2019, NeuroImage), reducing classifier
        # mismatch. Used as default in EEGLAB (Delorme & Makeig 2004).
        ica = run_ica(
            epochs_clean,
            n_components=N_COMPONENTS,
            method="infomax",
            fit_params=dict(extended=True),
            random_state=RANDOM_STATE,
        )

        ica.save(PREP_DIR / f"sub-{subject}_ica.fif", overwrite=True)

        save_ica_component_plots(ica, PREP_DIR, subject)

        # --- QC: preprocessing ---
        n_orig_epochs  = len(epochs)
        n_clean_epochs = len(epochs_clean)
        n_bad_channels = len(bads)
        concat_dur_s   = raw_concat.times[-1]
        n_ica_comps    = ica.n_components_

        if n_clean_epochs < 20 or n_ica_comps < 10:
            prep_flag = "fail"
        elif n_clean_epochs < 40 or n_bad_channels > 5:
            prep_flag = "warn"
        else:
            prep_flag = "pass"

        log_qc(
            qc_dir  = QC_DIR,
            subject = subject,
            stage   = "preprocessing",
            flag    = prep_flag,
            metrics = {
                "concat_dur_s":   round(float(concat_dur_s), 1),
                "n_bad_channels": n_bad_channels,
                "n_orig_epochs":  n_orig_epochs,
                "n_clean_epochs": n_clean_epochs,
                "n_ica_comps":    n_ica_comps,
            },
        )
        print(f"  QC preprocessing: {prep_flag}  "
              f"dur={concat_dur_s:.0f}s  bads={n_bad_channels}  "
              f"epochs={n_clean_epochs}/{n_orig_epochs}  "
              f"ica_comps={n_ica_comps}")

    except FileNotFoundError as e:
        print(f"\n  [SKIP] sub-{subject}: file not found -- {e}")
        continue
    except Exception as e:
        print(f"\n  [ERROR] sub-{subject}: unexpected error -- {e}")
        import traceback
        traceback.print_exc()
        continue

print("\nICA FIT COMPLETE")
print(f"\nDone. Processed {len(SUBJECTS)} subject(s): {SUBJECTS}")
