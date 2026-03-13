"""
data_preica.py
Pre-ICA data preparation: detect and interpolate bad channels, re-reference, create epochs, and apply autoreject.
"""

from pathlib import Path
from src.config import DIR_DATA, DIR_SIGCLEAN
from src.preprocessing import (
    load_clean_raw,
    detect_bad_channels,
    interpolate_bad_channels,
    rereference_raw,
    create_fixed_length_epochs,
    run_autoreject,
    save_epochs,
)

BAD_CHAN_THRESHOLD = 3.0
EPOCH_DURATION = 5.0  # seconds
SUBJECTS = ["S18"]
INPUT_DIR = DIR_SIGCLEAN
OUTPUT_DIR = DIR_DATA / "d03_pre_ica"

for subject in SUBJECTS:
    print(f"\nProcessing subject {subject}")
    raw = load_clean_raw(subject, INPUT_DIR)

    # Detect bad channels
    print("Detecting bad channels")
    raw, bads = detect_bad_channels(raw, BAD_CHAN_THRESHOLD)
    
    # Interpolate bad channels
    raw = interpolate_bad_channels(raw)

    # Re-reference to common average
    raw = rereference_raw(raw)

    # Create fixed length epochs 
    epochs = create_fixed_length_epochs(raw, EPOCH_DURATION)

    # Run autoreject on epoched data

    epochs_ar, reject_log = run_autoreject(epochs)
    
    save_epochs(epochs_ar, OUTPUT_DIR, subject)