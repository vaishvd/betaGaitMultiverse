"""
Shared preprocessing functions for betaGaitMultiverse.

Used by both the canonical pipeline (prepana scripts) and the
multiverse pipeline (multiverse_pipeline.py). Centralising this logic
ensures both pipelines apply identical preprocessing for a given
decision set.
"""

import numpy as np
import mne
from pathlib import Path
from autoreject import AutoReject

from src.paths import get_dataset_dirs
from src.preprocessing import (
    drop_invalid_channels,
    drop_invalid_eeg_channels,
)
from src.ica_utils import run_ica, label_and_mark_ica
from src.nodes.asr_node import apply_asr_node

TARGET_SFREQ = 250
LINE_FREQ    = 50.0
EPOCH_DUR    = 2.0
N_COMPONENTS = 0.99
RANDOM_STATE = 42
AMP_THRESH   = 350e-6


def load_and_concatenate(
    subject: str,
    raw_dir: Path,
) -> mne.io.BaseRaw:
    """
    Load STAND and CS EEG recordings, concatenate, and add annotations.

    Loads both recordings, picks EEG channels, sets standard_1005
    montage, verifies matching channel names and sfreq, concatenates
    with STAND first, and appends STAND/CS annotations.

    Parameters
    ----------
    subject : str
    raw_dir : Path to d00_raw/

    Returns
    -------
    mne.io.BaseRaw  preloaded concatenated raw, unreferenced, unfiltered
    """
    stand_vhdr = raw_dir / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-STAND.vhdr"
    walk_vhdr  = raw_dir / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-CS.vhdr"
    for p in (stand_vhdr, walk_vhdr):
        if not p.exists():
            raise FileNotFoundError(p)

    def _load(path):
        r = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        drop_invalid_channels(r)
        r.pick("eeg")
        r.set_montage("standard_1005", on_missing="ignore")
        return r

    raw_stand = _load(stand_vhdr)
    raw_walk  = _load(walk_vhdr)

    if raw_stand.ch_names != raw_walk.ch_names:
        raise RuntimeError(
            f"sub-{subject}: STAND and CS channel names differ"
        )
    if raw_stand.info["sfreq"] != raw_walk.info["sfreq"]:
        raise RuntimeError(
            f"sub-{subject}: STAND and CS sfreq differ"
        )

    n_stand   = raw_stand.n_times
    sfreq0    = raw_stand.info["sfreq"]
    stand_dur = n_stand / sfreq0
    walk_dur  = raw_walk.n_times / sfreq0

    raw_concat = mne.concatenate_raws([raw_stand, raw_walk], preload=True)
    raw_concat.annotations.append(0.0,        stand_dur, "STAND")
    raw_concat.annotations.append(stand_dur,  walk_dur,  "CS")

    print(f"  sub-{subject}: concat {raw_concat.n_times} samples  "
          f"sfreq={raw_concat.info['sfreq']} Hz")
    return raw_concat


def preprocess_raw(
    raw: mne.io.BaseRaw,
    subject: str,
    highpass_hz: float = 1.0,
    lowpass_hz: float | None = 60.0,
    use_asr: bool = False,
    asr_cutoff: float = 30.0,
) -> mne.io.BaseRaw:
    """
    Apply filtering, bad-channel interpolation, optional ASR,
    and average reference to a concatenated raw recording.

    Parameters
    ----------
    raw         : preloaded concatenated raw (STAND + CS)
    subject     : subject id for logging
    highpass_hz : high-pass filter cutoff in Hz
    lowpass_hz  : low-pass filter cutoff in Hz, or None to skip
    use_asr     : whether to apply ASR before referencing
    asr_cutoff  : ASR SD threshold (default 30, Gorjan et al. 2022)

    Returns
    -------
    mne.io.BaseRaw  filtered, interpolated, referenced raw
    """
    if raw.info["sfreq"] > TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    raw.filter(l_freq=float(highpass_hz), h_freq=None,
               fir_design="firwin", verbose=False)
    if lowpass_hz is not None:
        raw.filter(l_freq=None, h_freq=float(lowpass_hz),
                   fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=LINE_FREQ, verbose=False)

    data = raw.get_data()
    ptp  = np.ptp(data, axis=1)
    z    = (ptp - ptp.mean()) / (ptp.std() + 1e-12)
    bads = [raw.ch_names[i] for i in np.where(np.abs(z) > 3.0)[0]]
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True)
        print(f"  sub-{subject}: bad channels {bads}")
    else:
        print(f"  sub-{subject}: bad channels []")

    if use_asr:
        stand_ann = [
            a for a in raw.annotations if a["description"] == "STAND"
        ][0]
        calib = raw.copy().crop(
            stand_ann["onset"],
            min(stand_ann["onset"] + stand_ann["duration"],
                raw.times[-1])
        )
        calib = calib.crop(tmax=calib.times[-1] - 2.0)
        raw = apply_asr_node(raw, apply=True,
                             calib_raw=calib, cutoff=asr_cutoff)
        print(f"  sub-{subject}: ASR applied (cutoff={asr_cutoff})")
    else:
        print(f"  sub-{subject}: ASR skipped")

    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def fit_ica(
    raw: mne.io.BaseRaw,
    subject: str,
    brain_thresh: float = 0.7,
    ica_path: Path | None = None,
    iclean_path: Path | None = None,
    epo_path: Path | None = None,
) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA, int]:
    """
    Fit Extended Infomax ICA on fixed-length epochs, run ICLabel,
    apply to raw, and optionally save to disk.

    Checks for cached ICA files first — if both ica_path and
    iclean_path exist, loads from cache instead of recomputing.

    Parameters
    ----------
    raw          : preprocessed, referenced raw
    subject      : subject id for logging
    brain_thresh : ICLabel brain probability threshold
    ica_path     : optional path to save/load ICA object (.fif)
    iclean_path  : optional path to save/load cleaned raw (.fif)

    Returns
    -------
    raw_clean  : ICA-cleaned raw
    ica        : fitted ICA object with exclusions set
    n_brain    : number of retained brain components
    """
    if (ica_path is not None and iclean_path is not None
            and ica_path.exists() and iclean_path.exists()):
        print(f"  sub-{subject}: loading cached ICA")
        raw_clean = mne.io.read_raw_fif(
            iclean_path, preload=True, verbose=False
        )
        ica = mne.preprocessing.read_ica(ica_path)
        n_brain = ica.n_components_ - len(ica.exclude)
        return raw_clean, ica, n_brain

    epochs_raw = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_DUR, preload=True, verbose=False
    )
    epochs_raw.pick("eeg")
    drop_invalid_eeg_channels(epochs_raw)

    ar = AutoReject(
        n_interpolate=[1, 2, 4],
        random_state=RANDOM_STATE,
        verbose=False,
    )
    ar.fit(epochs_raw)
    epochs_clean, _ = ar.transform(epochs_raw, return_log=True)

    if len(epochs_clean) < 20:
        raise RuntimeError(
            f"sub-{subject}: only {len(epochs_clean)} clean epochs"
        )
    print(f"  sub-{subject}: {len(epochs_clean)} clean epochs")

    if epo_path is not None:
        epochs_clean.save(epo_path, overwrite=True)

    ica = run_ica(
        epochs_clean,
        n_components=N_COMPONENTS,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=RANDOM_STATE,
    )

    result = label_and_mark_ica(
        ica, epochs_clean, brain_thresh=float(brain_thresh)
    )
    n_brain = len(result["brain_ics"])
    if n_brain == 0:
        raise RuntimeError(f"sub-{subject}: no brain ICs retained")
    print(f"  sub-{subject}: {n_brain} brain ICs  "
          f"{len(ica.exclude)} excluded")

    ica.apply(raw)

    pos = np.array([
        raw.info["chs"][i]["loc"][:3]
        for i in range(len(raw.ch_names))
    ])
    no_pos = [
        raw.ch_names[i]
        for i in range(len(raw.ch_names))
        if not np.any(pos[i] != 0)
    ]
    if no_pos:
        raw.drop_channels(no_pos)
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True)

    if ica_path is not None:
        ica.save(ica_path, overwrite=True)
    if iclean_path is not None:
        raw.save(iclean_path, overwrite=True)
        print(f"  sub-{subject}: ICA saved to {iclean_path.parent.name}/")

    return raw, ica, n_brain


def apply_ica(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    subject: str,
    iclean_path: Path | None = None,
) -> mne.io.BaseRaw:
    """
    Apply a pre-fitted ICA (with exclusions already set) to raw data.

    Drops channels with missing electrode positions after ICA,
    interpolates any remaining bad channels, and optionally saves
    the cleaned raw to disk.

    Called by prepana03 in the canonical pipeline after visual ICA
    inspection. The multiverse pipeline uses fit_ica() which combines
    fitting and application in one step.

    Parameters
    ----------
    raw         : preprocessed raw (filtered, referenced)
    ica         : fitted ICA object with ica.exclude already set
    subject     : subject id for logging
    iclean_path : optional path to save cleaned raw (.fif)

    Returns
    -------
    mne.io.BaseRaw  ICA-cleaned raw
    """
    ica.apply(raw)

    pos = np.array([
        raw.info["chs"][i]["loc"][:3]
        for i in range(len(raw.ch_names))
    ])
    no_pos = [
        raw.ch_names[i]
        for i in range(len(raw.ch_names))
        if not np.any(pos[i] != 0)
    ]
    if no_pos:
        raw.drop_channels(no_pos)

    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True)

    if iclean_path is not None:
        raw.save(iclean_path, overwrite=True)
        print(f"  sub-{subject}: cleaned raw saved -> "
              f"{iclean_path.name}")

    return raw
