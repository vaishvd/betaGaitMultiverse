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
from src.config import DIR_DATA, DIR_SEG
from src.preprocessing import prepare_eeg_channels

L_FREQ = 1.0

INPUT_DIR = DIR_SEG
OUTPUT_DIR = DIR_DATA / "d02_sigclean"
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
    raw.filter(l_freq=L_FREQ, h_freq=None)

    print("Removing line noise")
    raw.notch_filter(freqs= (60, 100))

    output_file = OUTPUT_DIR / f"sub-{subject}_clean_raw.fif"

    raw.save(output_file, overwrite=True)

    print("Saved cleaned data →", output_file)