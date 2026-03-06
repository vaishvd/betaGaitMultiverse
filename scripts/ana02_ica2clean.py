"""ana02_ica2clean.py

Apply ICLabel to ICA solution, keep only brain components, and save cleaned data.

Steps:
1. Load raw EEG data (BIDS format)
2. Pick EEG channels, filter, detect/interpolate bad channels, re-reference
3. Load ICA solution
4. Run ICLabel to classify components
5. Exclude non-brain components
6. Apply ICA to raw data
7. Save cleaned data
"""

import mne
import mne_bids
import numpy as np
from mne_icalabel import label_components

from src.config import DIR_DATA, DIR_ICA, DIR_CLEAN, SUBJECTS

# Parameters (must match ana01)
L_FREQ = 1.0
BAD_CHAN_THRESHOLD = 3.0

for subject in SUBJECTS:
    print(f"\n{'=' * 60}")
    print(f"Processing sub-{subject}")
    print(f"{'=' * 60}")

    # --- Step 1: Load raw data from BIDS ---
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        task="task",
        datatype="eeg",
        root=DIR_DATA,
    )
    raw = mne_bids.read_raw_bids(bids_path, verbose="WARNING")

    # --- Step 2: Reproduce preprocessing (must match ana01) ---
    raw.pick("eeg")
    raw.load_data()
    raw.filter(l_freq=L_FREQ, h_freq=None)

    data = raw.get_data()
    ch_std = np.std(data, axis=1)
    median_std = np.median(ch_std)
    mad_std = np.median(np.abs(ch_std - median_std))
    bad_chans = [
        raw.ch_names[i]
        for i, s in enumerate(ch_std)
        if np.abs(s - median_std) > BAD_CHAN_THRESHOLD * mad_std
    ]
    raw.info["bads"] = bad_chans
    if bad_chans:
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average")

    # --- Step 3: Load ICA solution ---
    ica_fname = DIR_ICA / f"sub-{subject}_task-task_ica.fif"
    ica = mne.preprocessing.read_ica(ica_fname)

    # --- Step 4: Run ICLabel ---
    labels = label_components(raw, ica, method="iclabel")
    print(f"  ICLabel predictions: {dict(zip(labels['labels'], [f'{p:.2f}' for p in labels['y_pred_proba']]))}")

    # --- Step 5: Exclude non-brain components ---
    ica.exclude = [
        i for i, label in enumerate(labels["labels"]) if label != "brain"
    ]
    print(f"  Excluding {len(ica.exclude)}/{ica.n_components_} components: {ica.exclude}")

    # --- Step 6: Apply ICA ---
    ica.apply(raw)

    # --- Step 7: Save cleaned data ---
    clean_fname = DIR_CLEAN / f"sub-{subject}_task-task_desc-clean_raw.fif"
    raw.save(clean_fname, overwrite=True)
    print(f"  Saved: {clean_fname.name}")

print("\nDone.")
