from pathlib import Path
import numpy as np
import mne
from mne_icalabel import label_components

def load_epochs_ica(subject: str, epoch_file: Path):
    """Load preprocessed epochs for ICA."""
    print(f"Loading epochs for {subject}")
    epochs = mne.read_epochs(epoch_file, preload=True)

    # Ensure only EEG channels
    epochs.pick("eeg")

    return epochs

def load_raw_for_ica(subject: str, dir_sigclean: Path):
    """Load continuous data to apply ICA."""
    fname = dir_sigclean / f"sub-{subject}_clean_raw.fif"
    raw = mne.io.read_raw_fif(fname, preload=True)
    return raw

def run_ica(epochs: mne.Epochs, method="fastica", n_components=0, random_state=42):
    """Fit ICA to epoched data."""
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state
    )
    rank = mne.compute_rank(epochs)
    print("Estimated rank:", rank)
    ica.fit(epochs, decim=3)
    ica.plot_components(title="ICA Components")

    return ica

def apply_iclabel(ica: mne.preprocessing.ICA, epochs: mne.Epochs):
    """Run ICLabel and exclude non-brain components."""
    labels = label_components(epochs, ica, method="iclabel")

    print(f"ICLabel predictions: {labels['labels']}")

    ica.exclude = [
        i for i, label in enumerate(labels["labels"])
        if label != "brain"
    ]

    print(f"Excluding {len(ica.exclude)} components: {ica.exclude}")

    if ica.exclude:
        ica.plot_components(picks=ica.exclude, title="Excluded Components")

    epochs_clean = ica.apply(epochs.copy())

    return epochs_clean, ica