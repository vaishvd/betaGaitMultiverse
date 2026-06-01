"""
ana03_ica2clean.py
==================
Classify ICA components with ICLabel, reject artefact ICs,
and apply the cleaned decomposition to the continuous raw EEG.

Input
-----
d02_prep/
    sub-{sub}_clean_raw.fif          preprocessed continuous raw
    sub-{sub}_ica.fif                fitted ICA (no exclusions)
    sub-{sub}_preica_clean_epo.fif   clean epochs (needed for ICLabel features)

Output
------
d03_clean/
    sub-{sub}_desc-icaClean_raw.fif  ICA-cleaned continuous raw
"""

import mne
import numpy as np

from src.paths import get_dataset_dirs
from src.ica_utils import label_and_mark_ica

DATASET  = "stepup"
SUBJECTS = ["S1"]

# Minimum probability for a component to be accepted as brain.
# Components below this threshold (or labelled as non-brain) are excluded.
BRAIN_THRESH = 0.7

dirs      = get_dataset_dirs(DATASET)
PREP_DIR  = dirs["prep"]
CLEAN_DIR = dirs["clean"]

for subject in SUBJECTS:

    print(f"\n{'='*60}")
    print(f"ICA CLEANING: sub-{subject}")
    print(f"{'='*60}")

    # Load preprocessed continuous raw 

    raw = mne.io.read_raw_fif(
        PREP_DIR / f"sub-{subject}_clean_raw.fif",
        preload=True, verbose=False,
    )

    nan_frac = np.mean(np.isnan(raw.get_data()))
    print(f"  Input NaN fraction : {nan_frac*100:.1f}%")
    if nan_frac > 0.01:
        print(f"  ERROR: clean_raw.fif is corrupt ({nan_frac*100:.0f}% NaN)."
              f" Re-run ana02_raw2ica.py to regenerate it.")
        continue

    print(f"  sfreq    : {raw.info['sfreq']:.0f} Hz")
    print(f"  duration : {raw.times[-1]:.1f} s")
    print(f"  channels : {len(raw.ch_names)}")

    # Load ICA 

    ica = mne.preprocessing.read_ica(
        PREP_DIR / f"sub-{subject}_ica.fif",
        verbose=False,
    )
    print(f"\n  ICA components : {ica.n_components_}")
    print(f"  Excluded on load: {ica.exclude}  "
          f"(should be empty — exclusion happens here)")

    # Load pre-ICA epochs 

    epochs = mne.read_epochs(
        PREP_DIR / f"sub-{subject}_preica_clean_epo.fif",
        preload=True, verbose=False,
    )
    print(f"  Epochs for ICLabel: {len(epochs)} × {len(epochs.ch_names)} ch")

    # IC classification and rejection 

    result = label_and_mark_ica(ica, epochs, brain_thresh=BRAIN_THRESH)

    print(f"\n  Brain ICs kept   : {len(result['brain_ics'])}  "
          f"{result['brain_ics']}")
    print(f"  Artefact ICs out : {len(result['exclude_ics'])}  "
          f"{result['exclude_ics']}")

    if len(result["brain_ics"]) == 0:
        print("  ERROR: no brain ICs survived — check BRAIN_THRESH or "
              "inspect topographies manually.")
        continue

    if len(result["brain_ics"]) < 3:
        print("  WARNING: fewer than 3 brain ICs — consider lowering "
              "BRAIN_THRESH or reviewing components.")

    #  Save ICA with exclusions so ana05 can apply same cleaning to baseline

    ica_clean_out = CLEAN_DIR / f"sub-{subject}_ica-clean.fif"
    ica.save(ica_clean_out, overwrite=True, verbose=False)
    print(f"  Saved ICA with exclusions → {ica_clean_out.name}")

    # Apply ICA to continuous raw 

    raw_clean = ica.apply(raw.copy(), verbose=False)

    #  Drop channels without valid sensor positions 
    # standard_1020 montage doesn't cover every channel name; those channels
    # get NaN positions and break the interpolation spline matrix.

    no_pos = [
        ch["ch_name"] for ch in raw_clean.info["chs"]
        if ch["kind"] == 2 and (                         # EEG channel
            np.any(np.isnan(ch["loc"][:3])) or
            np.allclose(ch["loc"][:3], 0)
        )
    ]
    if no_pos:
        print(f"\n  Dropping {len(no_pos)} channels with no sensor position: {no_pos}")
        raw_clean.drop_channels(no_pos)

    # Interpolate bad channels 

    if raw_clean.info["bads"]:
        print(f"\n  Interpolating bad channels: {raw_clean.info['bads']}")
        raw_clean.interpolate_bads(reset_bads=True)

    # QC: verify no NaN introduced by ICA 

    nan_out = np.mean(np.isnan(raw_clean.get_data()))
    print(f"\n  Output NaN fraction: {nan_out*100:.1f}%")
    if nan_out > 0:
        print("  ERROR: NaN in ICA output — inspect component mixing matrix.")
        continue

    #  Save

    out = CLEAN_DIR / f"sub-{subject}_desc-icaClean_raw.fif"
    raw_clean.save(out, overwrite=True, verbose=False)
    print(f"  Saved → {out.name}")

print("\nICA COMPLETE")
