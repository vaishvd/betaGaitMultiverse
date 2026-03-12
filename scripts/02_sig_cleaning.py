"""
Pre-ICA signal cleaning.

This script executes the following preprocessing steps:
1. Load segmented data
2. Select only EEG channels
3. Apply high-pass filter
4. Remove line-noise
5. Save clean data

"""

from pathlib import Path
import mne
from src.config import DIR_DATA
from src.preprocessing import (
    prepare_eeg_channels,
    highpass_filter,
    notch_filter
)

L_FREQ = 1.0
BAD_CHAN_THRESHOLD = 3.0

INPUT_DIR = DIR_DATA / "segmented"
OUTPUT_DIR = DIR_DATA / "clean"
OUTPUT_DIR.mkdir(exist_ok=True)


# Run only one subject
SUBJECTS = ["S18"]


for subject in SUBJECTS:

    print(f"\nProcessing subject {subject}")

    raw_file = INPUT_DIR / f"sub-{subject}_preadapt_raw.fif"

    raw = mne.io.read_raw_fif(raw_file, preload=True)

    print("Preparing EEG channels")
    raw = prepare_eeg_channels(raw)

    print("Applying high-pass filter")
    raw = highpass_filter(raw, L_FREQ)

    print("Removing line noise")
    raw = notch_filter(raw)

    output_file = OUTPUT_DIR / f"sub-{subject}_clean_raw.fif"

    raw.save(output_file, overwrite=True)

    print("Saved cleaned data →", output_file)