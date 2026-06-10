"""
Preprocessing utilities for EEG data.
Covers every step from raw segmented data up to pre-ICA clean epochs.
"""

import numpy as np
import mne
from pathlib import Path


def drop_invalid_channels(raw):
    """Drop channels with missing or zero electrode positions."""

    invalid_labels = [
        "status",
        "counter",
        "counter 2power24",
        "source"
    ]

    to_drop = [
        ch for ch in raw.ch_names
        if any(bad in ch.lower() for bad in invalid_labels)
    ]

    if len(to_drop) > 0:
        print(f"Dropping invalid channels: {to_drop}")
        raw.drop_channels(to_drop)
    else:
        print("No invalid channels found.")

    return raw


def drop_invalid_eeg_channels(epochs):
    """Drop EEG channels with NaN or flat (zero) location vectors."""

    bad_channels = []

    for ch in epochs.info["chs"]:
        if ch["kind"] == 2:  # EEG
            loc = ch["loc"][:3]

            if (
                np.any(np.isnan(loc)) or
                np.all(loc == 0.0)
            ):
                bad_channels.append(ch["ch_name"])

    if bad_channels:
        print(f"  Dropping {len(bad_channels)} invalid EEG channels: {bad_channels}")
        return epochs.drop_channels(bad_channels)
    return epochs


def save_epochs(epochs_ar: mne.Epochs, output_dir: Path, subject: str):
    """Save MNE Epochs object to FIF file and print confirmation."""

    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"sub-{subject}_preica_clean_epo.fif"
    epochs_ar.save(output_file, overwrite=True)
    print("Saved pre-ICA epochs ->", output_file)
    return output_file
