"""
Single-subject multiverse pipeline for betaGaitMultiverse.
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy.stats import ttest_rel

from src.config import DATASET, DIR_MULTIVERSE_BRANCHES
from src.paths import get_dataset_dirs
from src.pipeline_steps import load_and_concatenate, preprocess_raw, fit_ica
from src.spatial_filter import gaussian_roi_weights, apply_gaussian_roi

TARGET_SFREQ = 250
AMP_THRESH   = 350e-6
FREQS        = np.arange(13, 31, dtype=float)
N_CYCLES_WAV = FREQS / 2.0
N_POINTS     = 101
EDGE_CROP    = 0.05

# Peak-window boundaries (% of gait cycle → sample index in 101-point axis)
STANCE_PEAK_START = 5
STANCE_PEAK_END   = 25
SWING_PEAK_START  = 65
SWING_PEAK_END    = 85


def _branch_dir(subject: str, decisions: dict) -> Path:
    """Subdirectory keyed on ICA-relevant decisions only."""
    ica_keys = {"use_asr", "brain_thresh", "highpass_hz", "lowpass_hz"}
    parts = "_".join(
        f"{k}-{decisions[k]}"
        for k in sorted(ica_keys)
        if k in decisions
    )
    d = DIR_MULTIVERSE_BRANCHES / subject / parts
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_subject_multiverse(subject: str, decisions: dict) -> dict | None:
    """
    Run one analysis branch for one subject.

    Returns a dict with t_stat and supporting metrics, or None on failure.
    """
    dirs      = get_dataset_dirs(DATASET)
    raw_dir   = dirs["raw"]
    event_dir = dirs["gait_events"]

    branch_dir  = _branch_dir(subject, decisions)
    ica_path    = branch_dir / f"sub-{subject}_ica.fif"
    iclean_path = branch_dir / f"sub-{subject}_desc-icaClean_raw.fif"

    # Preprocessing 
    raw = load_and_concatenate(subject, raw_dir)
    raw = preprocess_raw(
        raw, subject,
        highpass_hz = float(decisions["highpass_hz"]),
        lowpass_hz  = decisions["lowpass_hz"],
        use_asr     = bool(decisions["use_asr"]),
        asr_cutoff  = 30.0,
    )
    raw_clean, ica, n_brain = fit_ica(
        raw, subject,
        brain_thresh = float(decisions["brain_thresh"]),
        ica_path     = ica_path,
        iclean_path  = iclean_path,
    )

    # Crop segments 
    def crop(r, desc):
        ann = [a for a in r.annotations if a["description"] == desc][0]
        return r.copy().crop(ann["onset"],
                             min(ann["onset"] + ann["duration"], r.times[-1]))

    raw_stand = crop(raw_clean, "STAND").crop(tmax=None)   # keep full
    raw_walk  = crop(raw_clean, "CS")
    sfreq     = raw_walk.info["sfreq"]

    # Standing baseline 
    if decisions["baseline_type"] == "standing":
        epochs = mne.make_fixed_length_epochs(
            raw_stand, duration=2.0, preload=True, verbose=False
        )
        tfrs = []
        for ep in epochs.get_data():
            if np.max(np.abs(ep)) > AMP_THRESH:
                continue
            pw = mne.time_frequency.tfr_array_morlet(
                ep[np.newaxis], sfreq=sfreq, freqs=FREQS,
                n_cycles=N_CYCLES_WAV, output="power",
                zero_mean=True, verbose=False
            )[0]
            c = int(EDGE_CROP * pw.shape[-1])
            tfrs.append(pw[..., c:-c] if c else pw)
        if not tfrs:
            raise RuntimeError(f"sub-{subject}: no clean standing epochs")
        baseline_power = np.stack(tfrs).mean(axis=(0, 3))   # (ch, freq)
    else:
        baseline_power = None   # computed from walking data

    #  Gait-cycle TFR 
    cycles    = pd.read_csv(event_dir / f"sub-{subject}_cycles.tsv", sep="\t")
    walk_data = raw_walk.get_data()

    tfr_cycles, rto_fracs = [], []
    for _, row in cycles.iterrows():
        i0 = int(round(row["rhs_start_s"] * sfreq))
        i1 = int(round(row["rhs_end_s"]   * sfreq))
        if i0 < 0 or i1 > walk_data.shape[1] or i1 <= i0:
            continue
        seg = walk_data[:, i0:i1]
        if not np.isfinite(seg).all() or np.max(np.abs(seg)) > AMP_THRESH:
            continue
        pw = mne.time_frequency.tfr_array_morlet(
            seg[np.newaxis], sfreq=sfreq, freqs=FREQS,
            n_cycles=N_CYCLES_WAV, output="power",
            zero_mean=True, verbose=False
        )[0]
        c = int(EDGE_CROP * pw.shape[-1])
        pw = pw[..., c:-c] if c else pw
        x_old = np.linspace(0, 1, pw.shape[-1])
        x_new = np.linspace(0, 1, N_POINTS)
        pw = np.array([[np.interp(x_new, x_old, pw[ch, f])
                        for f in range(pw.shape[1])]
                       for ch in range(pw.shape[0])])
        tfr_cycles.append(pw)
        dur = row["rhs_end_s"] - row["rhs_start_s"]
        rto_fracs.append(float(np.clip(
            (row["rto_s"] - row["rhs_start_s"]) / dur, 0.3, 0.8
        )))

    if len(tfr_cycles) < 20:
        raise RuntimeError(
            f"sub-{subject}: only {len(tfr_cycles)} gait cycles accepted"
        )
    tfr_stack = np.stack(tfr_cycles)   # (cycles, ch, freq, time)

    #  ERSP 
    if decisions["baseline_type"] == "standing":
        ersp = 10 * np.log10(
            tfr_stack / baseline_power[np.newaxis, :, :, np.newaxis]
        )
    else:
        baseline_walk = tfr_stack.mean(axis=(0, 3))
        ersp = 10 * np.log10(
            tfr_stack / baseline_walk[np.newaxis, :, :, np.newaxis]
        )

    # Gaussian spatial filter
    raw_ref = mne.io.read_raw_fif(iclean_path, preload=False, verbose=False)
    weights = gaussian_roi_weights(raw_ref.info, center_ch="Cz", sigma_mm=40.0)
    ersp_w  = np.stack([apply_gaussian_roi(ersp[k], weights)
                        for k in range(len(ersp))])   # (cycles, freq, time)

    # Phase split 
    rto_idx = np.round(
        np.array(rto_fracs) * (N_POINTS - 1)
    ).astype(int)

    if decisions["phase_window"] == "full":
        stance = np.array([ersp_w[k, :, :rto_idx[k]].mean()
                           for k in range(len(rto_idx))])
        swing  = np.array([ersp_w[k, :, rto_idx[k]:].mean()
                           for k in range(len(rto_idx))])
    else:   # "peak"
        def _idx(pct):
            return int(round(pct / 100 * (N_POINTS - 1)))
        stance = np.array([
            ersp_w[k, :, _idx(STANCE_PEAK_START):_idx(STANCE_PEAK_END)].mean()
            for k in range(len(rto_idx))
        ])
        swing  = np.array([
            ersp_w[k, :, _idx(SWING_PEAK_START):_idx(SWING_PEAK_END)].mean()
            for k in range(len(rto_idx))
        ])

    t_stat, t_pval = ttest_rel(stance, swing)
    print(f"  sub-{subject}: t={t_stat:.2f}  p={t_pval:.4f}  "
          f"stance={stance.mean():+.2f}  swing={swing.mean():+.2f}  "
          f"n_cycles={len(stance)}")

    return {
        "subject":          subject,
        "t_stat":           float(t_stat),
        "t_pval":           float(t_pval),
        "beta_stance_mean": float(stance.mean()),
        "beta_swing_mean":  float(swing.mean()),
        "n_cycles":         len(stance),
        "n_brain_ics":      int(n_brain),
        **decisions,
    }