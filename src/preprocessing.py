"""
Preprocessing utilities for EEG data.
Covers every step from raw segmented data up to pre-ICA clean epochs.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from autoreject import AutoReject
from pathlib import Path


def filter_raw(
    raw: mne.io.BaseRaw,
    target_sfreq: float = 512,
    l_freq: float = 1.0,
    line_freqs: list[float] | None = None,
) -> mne.io.BaseRaw:
    """
    Resample, high-pass filter, and notch filter, all in-place.

    Parameters
    ----------
    target_sfreq : resample target in Hz (skipped if already at target)
    l_freq       : high-pass cutoff in Hz
    line_freqs   : list of notch frequencies in Hz  (default: [50] for EU)
    """
    if line_freqs is None:
        line_freqs = [50]

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq)
        print(f"  Resampled        → {target_sfreq} Hz")

    raw.filter(l_freq=l_freq, h_freq=None, method="fir", fir_window="hamming")
    print(f"  High-pass filter @ {l_freq} Hz")

    raw.notch_filter(freqs=line_freqs)
    print(f"  Notch filter     @ {line_freqs} Hz")

    return raw


def save_sigclean_raw(
    raw: mne.io.BaseRaw,
    output_dir: Path,
    subject: str,
) -> Path:
    """
    Save filtered, bad-channel-corrected, re-referenced raw to the
    sigclean directory as sub-{subject}_clean_raw.fif.
    """
    out = output_dir / f"sub-{subject}_clean_raw.fif"
    raw.save(out, overwrite=True)
    print(f"  Saved sigclean   → {out.name}")
    return out


def save_psd_plot(
    raw: mne.io.BaseRaw,
    output_path: Path,
    fmax: float = 80,
) -> None:
    """Save a PSD plot as a QC figure (non-interactive)."""
    fig = raw.compute_psd(fmax=fmax).plot(show=False)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PSD plot         → {output_path.name}")


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

def rereference_raw(raw: mne.io.BaseRaw, ref_type: str = "average", plot: bool = True) -> mne.io.BaseRaw:
    """
    Apply EEG reference directly to data (projection=False) and optionally plot.
    """
    raw.set_eeg_reference(ref_type, projection=False)
    print(f"  Re-reference     : {ref_type}")
    if plot:
        raw.plot(scalings=dict(eeg=100e-6), title=f"EEG after {ref_type} reference")
    return raw

def apply_asr(
    raw: mne.io.BaseRaw,
    cutoff: float = 20.0,
) -> mne.io.BaseRaw:
    """
    Apply Artifact Subspace Reconstruction (ASR) to continuous raw EEG data.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous raw EEG data, already rereferenced. Must be preloaded.
    cutoff : float, optional
        ASR cutoff in standard deviations. Default is 20.0.
        Lower = more aggressive cleaning (recommended range: 15–25 for gait EEG).

    Returns
    -------
    raw : mne.io.BaseRaw
        Raw object with ASR applied.
    """
    from asrpy import ASR

    asr = ASR(sfreq=raw.info["sfreq"], cutoff=cutoff)
    asr.fit(raw)
    raw = asr.transform(raw)

    return raw

def create_fixed_length_epochs(raw: mne.io.BaseRaw, duration: float):
    """
    Create fixed-length epochs of given duration from raw data
    """

    events = mne.make_fixed_length_events(raw, id=1, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True)
    print(f"Created {len(epochs)} epochs of {duration} s each.")
    return epochs

def create_preica_epochs(raw, epoch_length):
    import mne
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=epoch_length,
        preload=True
    )
    return epochs

def run_autoreject(
    epochs: mne.Epochs,
    n_interpolate: list[int] = [1, 2, 3, 4],
    random_state: int = 11,
    plot: bool = False,
) -> tuple[mne.Epochs, object]:
    """
    Fit AutoReject on epochs and return (clean_epochs, reject_log).

    Parameters
    ----------
    plot : if True, display rejected epochs interactively (disable for batch runs)
    """
    ar = AutoReject(
        n_interpolate=n_interpolate,
        random_state=random_state,
        n_jobs=1,
        verbose=False,
    )
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    n_bad = int(reject_log.bad_epochs.sum())
    pct   = 100 * n_bad / len(epochs)
    print(f"  AutoReject       : {n_bad}/{len(epochs)} epochs rejected ({pct:.0f}%)")

    if pct > 30:
        print("  WARNING: >30% epochs rejected — inspect signal quality")

    if plot and reject_log.bad_epochs.any():
        epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6), title="Rejected Epochs")

    return epochs_ar, reject_log

def save_epochs(epochs_ar: mne.Epochs, output_dir: Path, subject: str):
    """
    Save pre-ICA epochs
    """

    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"sub-{subject}_preica_clean_epo.fif"
    epochs_ar.save(output_file, overwrite=True)
    print("Saved pre-ICA epochs →", output_file)
    return output_file