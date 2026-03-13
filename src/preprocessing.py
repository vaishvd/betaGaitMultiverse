"""
This file contains the functions required to run each preprocessing step on segmented data before ICA
"""

import numpy as np
import mne
from autoreject import AutoReject
from pathlib import Path


def prepare_eeg_channels(raw):
    """
    Select EEG channels and apply BioSemi128 montage.
    """

    raw.pick("eeg")
    raw.load_data()

    rename = {ch: ch.replace("1-", "", 1) for ch in raw.ch_names}
    raw.rename_channels(rename)

    montage = mne.channels.make_standard_montage("biosemi128")
    raw.set_montage(montage, on_missing="warn")

    return raw


def highpass_filter(raw, l_freq):
    """
    Apply high-pass filter before ICA.
    """
    raw.filter(l_freq=l_freq, h_freq=None)
    return raw


def notch_filter(raw, freqs=(50, 100)):
    """
    Remove line noise.
    """
    raw.notch_filter(freqs=freqs)
    return raw


def detect_bad_channels(raw, threshold=3.0):
    """
    Detect bad channels using variance outlier detection.
    """

    data = raw.get_data()

    chan_var = np.var(data, axis=1)
    zscores = (chan_var - chan_var.mean()) / chan_var.std()

    bad_idx = np.where(np.abs(zscores) > threshold)[0]
    bad_chs = [raw.ch_names[i] for i in bad_idx]

    raw.info["bads"].extend(bad_chs)

    return raw, bad_chs

def load_clean_raw(subject: str, input_dir: Path):
    """
    Load preprocessed raw FIF file for a subject
    """

    raw_file = input_dir / f"sub-{subject}_clean_raw.fif"
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    return raw

def interpolate_bad_channels(raw: mne.io.BaseRaw, plot=True):
    """
    Interpolate bad channels in the raw data and optionally plot them.
    """

    if raw.info["bads"]:
        print("Interpolating bad channels:", raw.info["bads"])
        if plot:
            raw.plot(scalings=dict(eeg=100e-6), title="Before Interpolation")
        raw.interpolate_bads(reset_bads=True)
        if plot:
            raw.plot(scalings=dict(eeg=100e-6), title="After Interpolation")
    else:
        print("No bad channels detected.")
    return raw

def rereference_raw(raw: mne.io.BaseRaw, ref_type="average", plot=True):
    """
    Set EEG reference and plot.
    """
    raw.set_eeg_reference(ref_type)
    print(f"Applied {ref_type} reference.")
    if plot:
        raw.plot(scalings=dict(eeg=100e-6), title=f"EEG after {ref_type} reference")
    return raw

def create_fixed_length_epochs(raw: mne.io.BaseRaw, duration: float):
    """
    Create fixed-length epochs of given duration from raw data
    """

    events = mne.make_fixed_length_events(raw, id=1, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True)
    print(f"Created {len(epochs)} epochs of {duration} s each.")
    return epochs

def run_autoreject(epochs: mne.Epochs, n_interpolate=[1, 2, 3, 4], random_state=11, plot = True):
    """
    Fit AutoReject and return clean epochs and reject log
    """

    ar = AutoReject(
        n_interpolate=n_interpolate,
        random_state=random_state,
        n_jobs=1,
        verbose=True
    )
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    print("Bad epochs detected:", reject_log.bad_epochs.sum())
    
    if plot and reject_log.bad_epochs.any():
        print("Plotting rejected epochs...")
        epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6), title="Rejected Epochs")

    return epochs_ar, reject_log

def save_epochs(epochs_ar: mne.Epochs, output_dir: Path, subject: str):
    """
    Save pre-ICA epochs
    """

    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"sub-{subject}_preica_epo_raw.fif"
    epochs_ar.save(output_file, overwrite=True)
    print("Saved pre-ICA epochs →", output_file)
    return output_file