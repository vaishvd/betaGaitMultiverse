"""
Shared preprocessing functions for betaGaitMultiverse.

Used by both the canonical pipeline (prepana scripts) and the
multiverse pipeline (multiverse_pipeline.py). Centralising this logic
ensures both pipelines apply identical preprocessing for a given
decision set.
"""

import numpy as np
import pandas as pd
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
from src.nodes.gedai_node import apply_gedai_node
from src.config import (
    TARGET_SFREQ, LINE_FREQ, BAD_CHANNEL_ZSCORE, RANDOM_STATE, N_COMPONENTS,
    ICA_METHOD, ICA_FIT_PARAMS, EPOCH_DUR, AUTOREJECT_N_INTERPOLATE,
    ASR_EDGE_TRIM_S, ASR_CALIBRATION_FLOOR_S, ICLABEL_RULE,
)

# Approximate adult head radius (metres), used to scale Jacobsen
# ds003039's unit-sphere electrode positions to a realistic physical
# size -- matches the ~0.095 m mean radius of MNE's own standard_1005
# montage (see _set_montage_from_electrodes_tsv).
HEAD_RADIUS_M = 0.095


def load_and_concatenate(
    subject: str,
    raw_dir: Path,
) -> mne.io.BaseRaw:
    """
    Load this dataset's raw EEG recording(s) into a single raw object
    annotated with "STAND" (quiet baseline segment) and "CS" (walking
    segment), branching on config.EEG_FORMAT:

      "brainvision" (stepUpAms)   : concatenate separate STAND.vhdr and
          CS.vhdr recordings (see _load_and_concatenate_brainvision).
      "eeglab" (Jacobsen ds003039): one continuous .set recording,
          annotated from its own events.tsv (see
          _load_and_annotate_eeglab).

    Downstream code (prepana02-07, multiverse_pipeline.py) crops on the
    "STAND"/"CS" annotation names only and never branches on dataset.

    Parameters
    ----------
    subject : str
    raw_dir : Path to d00_raw/

    Returns
    -------
    mne.io.BaseRaw  preloaded, EEG-only, montage set, unreferenced,
    unfiltered, with STAND/CS annotations.
    """
    from src.config import EEG_FORMAT

    if EEG_FORMAT == "brainvision":
        return _load_and_concatenate_brainvision(subject, raw_dir)
    elif EEG_FORMAT == "eeglab":
        return _load_and_annotate_eeglab(subject, raw_dir)
    raise ValueError(f"Unknown EEG_FORMAT: {EEG_FORMAT!r}")


def _load_and_concatenate_brainvision(
    subject: str,
    raw_dir: Path,
) -> mne.io.BaseRaw:
    """
    Load STAND and CS BrainVision recordings, concatenate, and add
    annotations. (stepUpAms path; unchanged from prior commits.)

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


def _set_montage_from_electrodes_tsv(
    raw: mne.io.BaseRaw,
    electrodes_path: Path,
    head_radius_m: float = HEAD_RADIUS_M,
) -> None:
    """
    Build and apply a DigMontage from a BIDS *_electrodes.tsv sidecar.

    ds003039's electrode positions are given on a unit sphere in
    EEGLAB's coordinate convention (X=anterior, Y=left, Z=superior),
    not MNE's head frame (X=right, Y=anterior, Z=superior) -- axes are
    permuted (mne_x=-eeglab_y, mne_y=eeglab_x, mne_z=eeglab_z) and
    scaled by `head_radius_m` to approximate a real head size. Verified
    against known channel geography (e.g. Fp1 anterior-left, O1/O2
    posterior) during the ds003039 inventory.

    Modifies `raw` in place (sets its montage); does not return a value.

    Parameters
    ----------
    raw             : mne.io.BaseRaw, channels already restricted to EEG
    electrodes_path : Path to the subject's *_electrodes.tsv
    head_radius_m   : float, optional. Scale factor for the unit-sphere
                      positions (default 0.095 m).
    """
    pos_df = pd.read_csv(electrodes_path, sep="\t")
    pos_df = pos_df[pos_df["name"].isin(raw.ch_names)]

    ch_pos = {}
    for _, row in pos_df.iterrows():
        x_eeglab, y_eeglab, z_eeglab = float(row["x"]), float(row["y"]), float(row["z"])
        ch_pos[row["name"]] = np.array([
            -y_eeglab * head_radius_m,
             x_eeglab * head_radius_m,
             z_eeglab * head_radius_m,
        ])

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(montage, on_missing="ignore")


def _load_and_annotate_eeglab(
    subject: str,
    raw_dir: Path,
) -> mne.io.BaseRaw:
    """
    Load one continuous EEGLAB recording (Jacobsen ds003039) and
    annotate it with the same "STAND"/"CS" segment names the shared
    pipeline crops on for stepUpAms, so prepana02-07 and
    multiverse_pipeline.py need no dataset-specific branching beyond
    this loader.

    "STAND" <- the config.BASELINE_START_VALUE .. BASELINE_END_VALUE
    event window (the dataset's own start_standing/end_standing quiet-
    standing block -- see config_jacobsen.py).
    "CS"    <- the config.SEGMENT_START_VALUE .. SEGMENT_END_VALUE
    event window (the non-button "easy" outdoor walking bout).

    Drops the 3 non-EEG MISC channels (x_dir/y_dir/z_dir head
    accelerometer axes) and sets the montage from the BIDS
    electrodes.tsv (see _set_montage_from_electrodes_tsv).

    Parameters
    ----------
    subject : str  e.g. "001"
    raw_dir : Path to d00_raw/

    Returns
    -------
    mne.io.BaseRaw  preloaded, EEG-only, montage set, unreferenced,
    unfiltered, with STAND/CS annotations.
    """
    from src.config import (
        TASK_NAME, SEGMENT_START_VALUE, SEGMENT_END_VALUE,
        BASELINE_START_VALUE, BASELINE_END_VALUE,
    )

    eeg_dir  = raw_dir / f"sub-{subject}" / "eeg"
    set_path = eeg_dir / f"sub-{subject}_task-{TASK_NAME}_eeg.set"
    if not set_path.exists():
        raise FileNotFoundError(set_path)

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    drop_invalid_channels(raw)

    # Drop non-EEG channels by type, read from channels.tsv (data-driven,
    # matches the resolve-from-channels.tsv convention used throughout
    # src.gait_cycles rather than hardcoding channel names).
    channels_path = eeg_dir / f"sub-{subject}_task-{TASK_NAME}_channels.tsv"
    ch_df = pd.read_csv(channels_path, sep="\t")
    misc_chs = [c for c in ch_df.loc[ch_df["type"] != "EEG", "name"] if c in raw.ch_names]
    if misc_chs:
        raw.drop_channels(misc_chs)

    # Also drop any channel present in the raw .set but NOT documented
    # in channels.tsv at all -- sub-018's raw file has 6 extra
    # Left/Right_Accelerometer_X/Y/Z channels absent from its own
    # channels.tsv (a BIDS metadata omission specific to that subject,
    # discovered when it crashed raw.interpolate_bads() in
    # preprocess_raw(): MNE's EEGLAB reader carried over nonzero
    # positions for 2 of them from the .set file's own chanlocs, which
    # is not caught by the type-based drop above since they're simply
    # absent from ch_df). Whitelisting to documented channels only
    # prevents this for this or any future subject with the same
    # omission.
    undocumented = [c for c in raw.ch_names if c not in set(ch_df["name"])]
    if undocumented:
        print(f"  sub-{subject}: dropping undocumented channels "
              f"(absent from channels.tsv): {undocumented}")
        raw.drop_channels(undocumented)

    electrodes_path = eeg_dir / f"sub-{subject}_task-{TASK_NAME}_electrodes.tsv"
    _set_montage_from_electrodes_tsv(raw, electrodes_path)

    events_path = eeg_dir / f"sub-{subject}_task-{TASK_NAME}_events.tsv"
    events = pd.read_csv(events_path, sep="\t")

    def _onset(value):
        rows = events.loc[events["value"] == value, "onset"]
        if len(rows) != 1:
            raise RuntimeError(
                f"sub-{subject}: expected exactly one '{value}' event in "
                f"{events_path.name}, found {len(rows)}"
            )
        return float(rows.iloc[0])

    walk_start = _onset(SEGMENT_START_VALUE)
    walk_end   = _onset(SEGMENT_END_VALUE)

    # Validated per-subject for all 18 analysis subjects (2026-08-10):
    # both BASELINE_START_VALUE ("start_standing") and BASELINE_END_VALUE
    # ("end_standing") are present exactly once, the resulting window is
    # exactly 240.0s for every subject, and it always falls strictly
    # before SEGMENT_START_VALUE (never overruns into the walking task).
    # This raises rather than silently coping -- if it ever fires for a
    # subject not covered by that validation, that's a real data problem
    # requiring a decision, not something to paper over.
    baseline_start = _onset(BASELINE_START_VALUE)
    baseline_end   = _onset(BASELINE_END_VALUE)
    if baseline_end <= baseline_start:
        raise RuntimeError(
            f"sub-{subject}: {BASELINE_END_VALUE} onset ({baseline_end:.1f}s) "
            f"is not after {BASELINE_START_VALUE} onset ({baseline_start:.1f}s)."
        )
    if baseline_end > raw.times[-1]:
        raise RuntimeError(
            f"sub-{subject}: {BASELINE_END_VALUE} onset ({baseline_end:.1f}s) "
            f"exceeds recording length ({raw.times[-1]:.1f}s)."
        )
    if baseline_end > walk_start:
        raise RuntimeError(
            f"sub-{subject}: baseline window [{baseline_start:.1f}, "
            f"{baseline_end:.1f}]s overruns {SEGMENT_START_VALUE} at "
            f"{walk_start:.1f}s -- needs an explicit decision."
        )

    raw.annotations.append(baseline_start, baseline_end - baseline_start, "STAND")
    raw.annotations.append(walk_start,     walk_end - walk_start,         "CS")

    print(f"  sub-{subject}: eeglab raw {raw.n_times} samples  "
          f"sfreq={raw.info['sfreq']:.0f} Hz  "
          f"STAND=[{baseline_start:.1f},{baseline_end:.1f}]s  "
          f"CS=[{walk_start:.1f},{walk_end:.1f}]s")
    return raw


def preprocess_raw(
    raw: mne.io.BaseRaw,
    subject: str,
    highpass_hz: float = 1.0,
    lowpass_hz: float | None = 60.0,
    use_asr: bool = False,
    asr_cutoff: float = 30.0,
    use_gedai: bool = False,
    gedai_noise_multiplier: float = 3.0,
) -> mne.io.BaseRaw:
    """
    Apply filtering, bad-channel interpolation, optional ASR,
    optional GEDAI, and average reference to a concatenated raw recording.

    Parameters
    ----------
    raw                    : preloaded concatenated raw (STAND + CS)
    subject                : subject id for logging
    highpass_hz            : high-pass filter cutoff in Hz
    lowpass_hz             : low-pass filter cutoff in Hz, or None to skip
    use_asr                : whether to apply ASR before referencing
    asr_cutoff             : ASR SD threshold (default 30, Gorjan et al. 2022)
    use_gedai              : whether to apply GEDAI after ASR
    gedai_noise_multiplier : GEDAI component rejection threshold (default 3.0)

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
    bads = [raw.ch_names[i] for i in np.where(np.abs(z) > BAD_CHANNEL_ZSCORE)[0]]
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
        calib = calib.crop(tmax=calib.times[-1] - ASR_EDGE_TRIM_S)
        calib_dur = calib.times[-1] - calib.times[0]
        print(f"  sub-{subject}: ASR calibration duration = "
              f"{calib_dur:.1f}s")
        # Floor lowered from 120.0 to 115.0 (2026-08-05, now
        # src.config.ASR_CALIBRATION_FLOOR_S): Jacobsen's validated
        # standing baseline (config_jacobsen.BASELINE_DURATION_S = 120.0s,
        # see results/pipeline/jacobsen/qc/baseline_120s_check.txt) is
        # always exactly 118.0s after the edge-safety trim above, for
        # every one of its 18 subjects -- deterministically 2s short of
        # the original 120.0 floor, so ASR could never calibrate on this
        # dataset at all. 115.0 comfortably admits Jacobsen's deterministic
        # 118.0s while still requiring ~2 minutes of calibration data.
        # stepUpAms is unaffected either way -- its calibration windows run
        # 180-187s per subject, far above both the old and new floor.
        if calib_dur < ASR_CALIBRATION_FLOOR_S:
            raise RuntimeError(
                f"sub-{subject}: ASR calibration too short "
                f"({calib_dur:.1f}s < {ASR_CALIBRATION_FLOOR_S:.0f}s required)"
            )
        raw = apply_asr_node(raw, apply=True,
                             calib_raw=calib, cutoff=asr_cutoff)
        print(f"  sub-{subject}: ASR applied (cutoff={asr_cutoff})")
    else:
        print(f"  sub-{subject}: ASR skipped")

    if use_gedai:
        raw = apply_gedai_node(
            raw,
            apply=True,
            broadband_noise_multiplier=gedai_noise_multiplier,
        )
        print(f"  sub-{subject}: GEDAI applied")
    else:
        print(f"  sub-{subject}: GEDAI skipped")

    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def fit_ica(
    raw: mne.io.BaseRaw,
    subject: str,
    iclabel_rule: str = ICLABEL_RULE,
    ica_path: Path | None = None,
    iclean_path: Path | None = None,
    epo_path: Path | None = None,
    ica_fit_highpass_hz: float | None = None,
    artifact_rule_fn=None,
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
    iclabel_rule : {"conservative", "balanced", "liberal"} -- passed to
                   src.ica_utils.select_ics_by_rule via label_and_mark_ica,
                   the same rule the multiverse pipeline uses. Ignored if
                   artifact_rule_fn is given.
    ica_path     : optional path to save/load ICA object (.fif)
    iclean_path  : optional path to save/load cleaned raw (.fif)
    ica_fit_highpass_hz : optional float. If given, AutoReject epoching
                   and the ICA fit itself run on a COPY of `raw`
                   additionally high-passed at this cutoff (e.g.
                   Jacobsen's paper-faithful ICA_bandpass_fmin=2.0Hz,
                   src.config_jacobsen.REFERENCE_ICA_FIT_HIGHPASS_HZ) --
                   the resulting ICA weights are then applied to the
                   ORIGINAL `raw` (at its own, gentler analysis
                   highpass), not to this copy. None (default) fits
                   directly on `raw`, unchanged behaviour.
    artifact_rule_fn : optional callable(probs) -> list[int]. If given,
                   used instead of src.ica_utils.select_ics_by_rule for
                   component exclusion (see label_and_mark_ica) -- e.g.
                   Jacobsen's paper-exact P(eye)>0.9 OR P(muscle)>0.9
                   artifact rule, kept deliberately decoupled from the
                   multiverse's shared iclabel_rule vocabulary.

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

    if ica_fit_highpass_hz is not None:
        raw_for_ica = raw.copy().filter(
            l_freq=float(ica_fit_highpass_hz), h_freq=None,
            fir_design="firwin", verbose=False,
        )
        print(f"  sub-{subject}: ICA fit on {ica_fit_highpass_hz:.1f}Hz-"
              f"highpassed copy (ICA_bandpass_fmin); weights applied "
              f"back to the analysis-highpass raw")
    else:
        raw_for_ica = raw

    epochs_raw = mne.make_fixed_length_epochs(
        raw_for_ica, duration=EPOCH_DUR, preload=True, verbose=False
    )
    epochs_raw.pick("eeg")
    drop_invalid_eeg_channels(epochs_raw)

    ar = AutoReject(
        n_interpolate=AUTOREJECT_N_INTERPOLATE,
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
        method=ICA_METHOD,
        fit_params=ICA_FIT_PARAMS,
        random_state=RANDOM_STATE,
    )

    result = label_and_mark_ica(
        ica, epochs_clean, rule=iclabel_rule, artifact_rule_fn=artifact_rule_fn,
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
    inspection, AND by multiverse_pipeline.run_subject_multiverse() (this
    function, unchanged, is the shared "apply this exclusion set" step
    for both pipelines -- see multiverse_pipeline.py's own docstring on
    _fit_or_load_ica). fit_ica() below is the reference-pipeline-only
    convenience wrapper that bundles fitting with a single application
    (a previous version of this docstring had this backwards).

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
        # Default fmt="single" (float32). The reference pipeline
        # (prepana04/05) reloads THIS file from disk for every
        # downstream stage, so it inherits the quantization;
        # run_subject_multiverse() keeps using the in-memory `raw`
        # returned below and never reloads this save -- see NOTES.md
        # for the ~0.9% reference-vs-universe_17 residual this causes
        # (confirmed benign, not fixed).
        raw.save(iclean_path, overwrite=True)
        print(f"  sub-{subject}: cleaned raw saved -> "
              f"{iclean_path.name}")

    return raw
