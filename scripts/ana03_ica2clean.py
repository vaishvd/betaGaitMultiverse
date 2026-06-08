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
    sub-{sub}_ica-clean.fif          ICA with exclusions (applied to baseline in ana05)
"""

import mne
import numpy as np

from src.paths import get_dataset_dirs
from src.ica_utils import label_and_mark_ica

DATASET  = "stepup"
SUBJECTS = ["S1"]

# Components below this threshold (or labelled as non-brain) are excluded.
BRAIN_THRESH = 0.7

dirs      = get_dataset_dirs(DATASET)
PREP_DIR  = dirs["prep"]
CLEAN_DIR = dirs["clean"]

for subject in SUBJECTS:

    print(f"ICA CLEANING: sub-{subject}")

    # Load preprocessed continuous raw 

    raw = mne.io.read_raw_fif(
        PREP_DIR / f"sub-{subject}_clean_raw.fif",
        preload=True, verbose=False,
    )

    nan_frac = np.mean(np.isnan(raw.get_data()))

    # Load ICA 

    ica = mne.preprocessing.read_ica(
        PREP_DIR / f"sub-{subject}_ica.fif",
        verbose=False,
    )
    print(f"\n  ICA components : {ica.n_components_}")

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

    #  Drop channels without valid sensor positions
    # get NaN positions and break the interpolation spline matrix.

    no_pos = [
        ch["ch_name"] for ch in raw.info["chs"]
        if ch["kind"] == 2 and (                         # EEG channel
            np.any(np.isnan(ch["loc"][:3])) or
            np.allclose(ch["loc"][:3], 0)
        )
    ]

    # Apply ICA to continuous raw
    raw_clean = ica.apply(raw.copy())

    if no_pos:
        raw_clean.drop_channels(no_pos)

    # Interpolate bad channels

    if raw_clean.info["bads"]:
        print(f"\n  Interpolating bad channels: {raw_clean.info['bads']}")
        raw_clean.interpolate_bads(reset_bads=True)

    # QC: verify no NaN introduced by ICA

    nan_out = np.mean(np.isnan(raw_clean.get_data()))
    if nan_out > 0:
        print("  ERROR: NaN in ICA output — inspect component mixing matrix.")
        continue

    #  Save ICA with exclusions so ana05 can apply same cleaning to baseline

    ica_clean_out = CLEAN_DIR / f"sub-{subject}_ica-clean.fif"
    ica.save(ica_clean_out, overwrite=True, verbose=False)
    print(f"  Saved ICA with exclusions → {ica_clean_out.name}")

    out = CLEAN_DIR / f"sub-{subject}_desc-icaClean_raw.fif"
    raw_clean.save(out, overwrite=True, verbose=False)
    print(f"  Saved → {out.name}")

print("\nICA COMPLETE")
