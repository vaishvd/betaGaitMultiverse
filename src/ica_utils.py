from pathlib import Path
import numpy as np
import mne
from mne_icalabel import label_components

def load_data_ica(subject: str, DATA_DIR: Path, l_freq=1.0, bad_chan_threshold=3.0):
    """Load raw EEG and reproduce pre-ICA preprocessing."""
    raw_file = DATA_DIR 
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    raw.pick("eeg")
    raw.load_data()
    raw.filter(l_freq=l_freq, h_freq=None)

    # Detect bad channels
    data = raw.get_data()
    ch_std = data.std(axis=1)
    median_std = np.median(ch_std)
    mad_std = np.median(abs(ch_std - median_std))
    bad_chans = [
        raw.ch_names[i]
        for i, s in enumerate(ch_std)
        if abs(s - median_std) > bad_chan_threshold * mad_std
    ]
    raw.info["bads"] = bad_chans
    if bad_chans:
        raw.plot(scalings=dict(eeg=100e-6), title=f"{subject} - Before Interpolation")
        raw.interpolate_bads(reset_bads=True)
        raw.plot(scalings=dict(eeg=100e-6), title=f"{subject} - After Interpolation")
    raw.set_eeg_reference("average")
    return raw

def run_ica(raw: mne.io.BaseRaw, method="fastica", n_components=0, random_state=42):
    """Fit ICA to raw data and return ICA object."""
    ica = mne.preprocessing.ICA(
        n_components=n_components, method=method, random_state=random_state
    )
    ica.fit(raw)
    ica.plot_components(title="ICA Components")
    return ica

def apply_iclabel(ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw):
    """Run ICLabel and exclude non-brain components, with visualization."""
    labels = label_components(raw, ica, method="iclabel")
    print(f"ICLabel predictions: {labels['labels']}")
    ica.exclude = [i for i, label in enumerate(labels["labels"]) if label != "brain"]
    print(f"Excluding {len(ica.exclude)} components: {ica.exclude}")
    
    if ica.exclude:
        ica.plot_components(picks=ica.exclude, title="Excluded Components")
    ica.apply(raw)
    return raw, ica