"""ana01_raw2ica.py

Preprocessing pipeline: Raw EEG → ICA decomposition.

Steps (following Jacobsen & Ferris, 2023, J Physiol, adapted for MNE):
1. Load raw EEG data (BIDS format)
2. Pick EEG channels only
3. High-pass filter at 1 Hz (FIR, zero-phase)
4. Detect and mark bad channels (> 3 SD of channel variance)
5. Interpolate bad channels
6. Re-reference to common average
7. Fit ICA (extended infomax)
8. Save ICA solution
"""

import mne
import mne_bids
import numpy as np

from src.config import DIR_RAWDATA, DIR_ICA

# Parameters
L_FREQ = 1.0
BAD_CHAN_THRESHOLD = 3.0  # standard deviations
ICA_METHOD = "infomax"

# find sub- folders in raw data directory and extract subject IDs
sub_dirs = [d for d in (DIR_RAWDATA).iterdir() if d.is_dir() and d.name.startswith("sub-")]

for sub in sub_dirs[0:3]:

    # --- Step 1: Load raw data from BIDS ---
    bids_path = mne_bids.BIDSPath(
        subject=sub.name.split("-")[1],
        task="task",
        datatype="eeg",
        root=DIR_RAWDATA,
    )
    raw = mne_bids.read_raw_bids(bids_path, verbose="WARNING")

    # --- Step 2: Pick EEG channels only ---
    raw.pick("eeg")
    raw.load_data()

    # Replace custom montage with standard BioSemi 128 montage (in head coords).
    # Channel names have a "1-" prefix that must be stripped to match biosemi128.
    rename = {ch: ch.replace("1-", "", 1) for ch in raw.ch_names}
    raw.rename_channels(rename)
    montage = mne.channels.make_standard_montage("biosemi128")
    raw.set_montage(montage, on_missing="warn")

    # --- Step 3: High-pass filter at 1 Hz ---
    raw.filter(l_freq=L_FREQ, h_freq=None)

    # --- Step 4: Detect bad channels (> 3 SD) ---
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
    print(f"  Bad channels ({len(bad_chans)}): {bad_chans}")

    # --- Step 5: Interpolate bad channels ---
    if bad_chans:
        raw.interpolate_bads(reset_bads=True)

    # --- Step 6: Re-reference to common average ---
    raw.set_eeg_reference("average")

    # raw.plot_sen

    # --- Step 7: Fit ICA ---
    ica = mne.preprocessing.ICA(
        method=ICA_METHOD,
        fit_params=dict(extended=True),
        random_state=42,
    )
    ica.fit(raw)
    print(f"  ICA components: {ica.n_components_}")

    # --- Step 8: Save ICA solution ---
    ica_fname = DIR_ICA / f"sub-{sub}_task-task_ica.fif"
    ica.save(ica_fname, overwrite=True)
    print(f"  Saved: {ica_fname.name}")

print("\nDone.")
