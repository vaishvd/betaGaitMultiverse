"""
Single-subject multiverse pipeline for betaGaitMultiverse.

Three decision nodes (highpass_hz, asr_mode, iclabel_rule). Computes the
same event-anchored double-stance-vs-swing beta contrast as the
canonical pipeline (prepana05_gaitcycles2tfr.py + prepana07_betaphase_
stats.py), reusing the shared warp/phase/beta-reduction logic from
src/ersp.py so both pipelines stay numerically identical when settings
match.
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from autoreject import AutoReject

from src.config import (
    DATASET, DIR_MULTIVERSE_BRANCHES, MULTIVERSE_TFR_FMAX,
    AMP_THRESH, TFR_FMIN, TFR_N_CYCLES_DIVISOR, TFR_N_POINTS, TFR_EDGE_CROP,
    EPOCH_DUR, N_COMPONENTS, RANDOM_STATE, ICA_METHOD, ICA_FIT_PARAMS,
    AUTOREJECT_N_INTERPOLATE, ASR_CUTOFF as REFERENCE_ASR_CUTOFF,
    ROI_CENTER_CH, BASELINE_EDGE_TRIM_S,
)
from src.paths import get_dataset_dirs
from src.preprocessing import drop_invalid_eeg_channels
from src.pipeline_steps import load_and_concatenate, preprocess_raw, apply_ica
from src.ica_utils import run_ica, iclabel_probabilities, select_ics_by_rule
from src.spatial_filter import linear_roi_weights
from src.ersp import (
    load_group_anchors,
    warp_cycle_to_grid,
    phase_split_indices,
    compute_standing_baseline,
    beta_roi_scalar,
)

# Re-exported under these names for backwards compatibility -- e.g.
# scripts/mulana04_zoom_universes.py imports FREQS/N_CYCLES_WAV/N_POINTS/
# EDGE_CROP/AMP_THRESH directly from this module.
FREQS        = np.arange(TFR_FMIN, int(MULTIVERSE_TFR_FMAX) + 1, dtype=float)  # 8-40 Hz, permanent
N_CYCLES_WAV = FREQS / TFR_N_CYCLES_DIVISOR
N_POINTS     = TFR_N_POINTS
EDGE_CROP    = TFR_EDGE_CROP

# ASR cutoff (SD threshold) per asr_mode. Higher cutoff = more lenient:
# asrpy/meegkit express the burst threshold directly as a multiple of the
# calibration data's clean-window SD, so a larger multiplier tolerates
# larger deviations before correcting them (see src/nodes/asr_node.py
# and Gorjan et al. 2022, who recommend 20-30 for walking EEG).
# "sd3" has no reference-pipeline equivalent (multiverse-only arm) and
# stays a literal; "sd20" is the same SD cutoff the reference pipeline
# now uses (src.config.ASR_CUTOFF), imported above rather than
# re-hardcoded, so the two can never drift apart again.
ASR_CUTOFF_BY_MODE = {"sd3": 3.0, "sd20": REFERENCE_ASR_CUTOFF}


def _branch_dir(subject: str, decisions: dict) -> Path:
    """
    Subdirectory keyed on ICA-relevant decisions only (highpass_hz,
    asr_mode, lowpass_hz). iclabel_rule is deliberately excluded: all
    three ICLabel rules are applied on top of the SAME cached ICA fit
    and ICLabel run, so this key gives 3 x 3 = 9 unique branches per
    subject instead of 27.
    """
    ica_keys = {"highpass_hz", "asr_mode", "lowpass_hz"}
    parts = "_".join(
        f"{k}-{decisions[k]}"
        for k in sorted(ica_keys)
        if k in decisions
    )
    d = DIR_MULTIVERSE_BRANCHES / subject / parts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fit_or_load_ica(raw, subject, branch_dir):
    """
    Fit ICA + run ICLabel once per (highpass_hz, asr_mode, lowpass_hz)
    branch, or load the cached fit/probabilities if already computed.

    Duplicates the AutoReject-epoching step from
    src.pipeline_steps.fit_ica() rather than calling it directly: that
    function bundles fitting with applying a single exclusion set and
    caches 1:1 on iclean_path, whereas here one ICA fit + ICLabel run
    must be reused across three different exclusion rules (see
    _branch_dir). The actual exclusion + apply step below reuses
    src.pipeline_steps.apply_ica() unchanged.

    Returns
    -------
    ica   : fitted mne.preprocessing.ICA (exclude not yet set)
    probs : ndarray, shape (n_components, 7) -- ICLabel class probabilities
    """
    ica_path   = branch_dir / f"sub-{subject}_ica.fif"
    probs_path = branch_dir / f"sub-{subject}_iclabel_probs.npy"

    if ica_path.exists() and probs_path.exists():
        ica   = mne.preprocessing.read_ica(ica_path, verbose=False)
        probs = np.load(probs_path)
        return ica, probs

    epochs_raw = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_DUR, preload=True, verbose=False
    )
    epochs_raw.pick("eeg")
    drop_invalid_eeg_channels(epochs_raw)

    ar = AutoReject(n_interpolate=AUTOREJECT_N_INTERPOLATE, random_state=RANDOM_STATE, verbose=False)
    ar.fit(epochs_raw)
    epochs_clean, _ = ar.transform(epochs_raw, return_log=True)

    if len(epochs_clean) < 20:
        raise RuntimeError(f"sub-{subject}: only {len(epochs_clean)} clean epochs")

    ica = run_ica(
        epochs_clean,
        n_components=N_COMPONENTS,
        method=ICA_METHOD,
        fit_params=ICA_FIT_PARAMS,
        random_state=RANDOM_STATE,
    )
    probs, _ = iclabel_probabilities(ica, epochs_clean)

    ica.save(ica_path, overwrite=True, verbose=False)
    np.save(probs_path, probs)
    return ica, probs


def run_subject_multiverse(subject: str, decisions: dict) -> dict | None:
    """
    Run one analysis branch for one subject.

    Parameters
    ----------
    decisions : dict with keys "highpass_hz" (float), "asr_mode"
        ("off"/"sd3"/"sd20"), "iclabel_rule" ("conservative"/"balanced"/
        "liberal"), "lowpass_hz" (float, fixed by the multiverse template
        to src.config.MULTIVERSE_LOWPASS_HZ = 40.0, permanently; not
        itself a forking decision node).

    Returns
    -------
    dict with the per-subject double-stance/swing beta contrast, or
    raises on failure (callers should catch and skip the subject).
    """
    dirs      = get_dataset_dirs(DATASET)
    raw_dir   = dirs["raw"]
    event_dir = dirs["gait_events"]
    ersp_dir  = dirs["ersp"]

    branch_dir  = _branch_dir(subject, decisions)
    iclabel_rule = decisions["iclabel_rule"]
    iclean_path  = branch_dir / f"sub-{subject}_desc-icaClean_{iclabel_rule}_raw.fif"

    # Preprocessing -- identical call to the canonical pipeline
    asr_mode   = decisions["asr_mode"]
    use_asr    = asr_mode != "off"
    asr_cutoff = ASR_CUTOFF_BY_MODE.get(asr_mode, 30.0)

    raw = load_and_concatenate(subject, raw_dir)
    raw = preprocess_raw(
        raw, subject,
        highpass_hz = float(decisions["highpass_hz"]),
        lowpass_hz  = float(decisions["lowpass_hz"]),
        use_asr     = use_asr,
        asr_cutoff  = asr_cutoff,
        use_gedai   = False,
    )

    # ICA: fit once per (highpass_hz, asr_mode, lowpass_hz) branch, cached;
    # apply this universe's iclabel_rule on top without refitting.
    ica, probs = _fit_or_load_ica(raw, subject, branch_dir)
    exclude_ics = select_ics_by_rule(probs, iclabel_rule)
    ica.exclude = exclude_ics
    n_brain = ica.n_components_ - len(exclude_ics)

    raw_clean = apply_ica(raw, ica, subject, iclean_path=iclean_path)

    # Crop segments
    def crop(r, desc):
        ann = [a for a in r.annotations if a["description"] == desc][0]
        return r.copy().crop(ann["onset"],
                             min(ann["onset"] + ann["duration"], r.times[-1]))

    raw_stand = crop(raw_clean, "STAND")
    stand_tmax = raw_stand.times[-1] - BASELINE_EDGE_TRIM_S   # trim boundary artefact, matches prepana05
    if stand_tmax <= 0:
        raise RuntimeError(
            f"sub-{subject}: standing segment too short after trimming "
            f"({raw_stand.times[-1]:.1f} s)"
        )
    raw_stand = raw_stand.crop(tmax=stand_tmax)

    raw_walk = crop(raw_clean, "CS")
    sfreq    = raw_walk.info["sfreq"]

    # Standing baseline -- fixed (canonical pipeline uses standing only;
    # baseline_type is no longer a multiverse decision node)
    baseline_power = compute_standing_baseline(
        raw_stand, FREQS, N_CYCLES_WAV,
        edge_crop=EDGE_CROP, amp_thresh=AMP_THRESH,
    )

    # Group-median gait-event anchors -- shared with the canonical pipeline
    A_lto, A_lhs, A_rto = load_group_anchors(ersp_dir)

    # Gait-cycle TFR + full-event (4-segment) time-warp
    cycles    = pd.read_csv(event_dir / f"sub-{subject}_cycles.tsv", sep="\t")
    walk_data = raw_walk.get_data()

    tfr_cycles = []
    for _, row in cycles.iterrows():
        i0 = int(round(row["rhs_start_s"] * sfreq))
        i1 = int(round(row["rhs_end_s"]   * sfreq))
        if i0 < 0 or i1 > walk_data.shape[1] or i1 <= i0:
            continue
        seg = walk_data[:, i0:i1]
        if not np.isfinite(seg).all() or np.max(np.abs(seg)) > AMP_THRESH:
            continue

        power = mne.time_frequency.tfr_array_morlet(
            seg[np.newaxis], sfreq=sfreq, freqs=FREQS,
            n_cycles=N_CYCLES_WAV, output="power",
            zero_mean=True, verbose=False,
        )[0]
        crop_n = int(EDGE_CROP * power.shape[-1])
        power = power[..., crop_n:-crop_n] if crop_n else power

        lto_idx = int(round(row["lto_s"] * sfreq)) - i0 - crop_n
        lhs_idx = int(round(row["lhs_s"] * sfreq)) - i0 - crop_n
        rto_idx = int(round(row["rto_s"] * sfreq)) - i0 - crop_n

        try:
            warped = warp_cycle_to_grid(
                power, lto_idx, lhs_idx, rto_idx,
                anchors=(A_lto, A_lhs, A_rto), n_points=N_POINTS,
            )
        except ValueError:
            continue

        tfr_cycles.append(warped)

    if len(tfr_cycles) < 20:
        raise RuntimeError(
            f"sub-{subject}: only {len(tfr_cycles)} gait cycles accepted"
        )
    tfr_stack = np.stack(tfr_cycles)   # (n_cycles, n_ch, n_freqs, N_POINTS)

    # ERSP: per-cycle log ratio to standing baseline
    ersp_per_cycle = 10 * np.log10(
        tfr_stack / baseline_power[np.newaxis, :, :, np.newaxis]
    )

    # Event-anchored double-stance / swing phase split (same anchors used
    # for warping above, so this is a fixed grid split -- see src.ersp)
    double_stance_idx, swing_idx = phase_split_indices((A_lto, A_lhs, A_rto), N_POINTS)

    ersp_double_stance = ersp_per_cycle[:, :, :, double_stance_idx].mean(axis=(0, -1))  # (n_ch, n_freqs)
    ersp_swing         = ersp_per_cycle[:, :, :, swing_idx].mean(axis=(0, -1))          # (n_ch, n_freqs)

    # Linear Cz-ROI beta-band reduction -- same as prepana07
    weights = linear_roi_weights(raw_clean.info, center_ch=ROI_CENTER_CH)
    beta_double_stance = beta_roi_scalar(ersp_double_stance, weights, FREQS)
    beta_swing          = beta_roi_scalar(ersp_swing,         weights, FREQS)

    print(f"  sub-{subject}: double_stance={beta_double_stance:+.2f} dB  "
          f"swing={beta_swing:+.2f} dB  diff={beta_double_stance - beta_swing:+.2f} dB  "
          f"n_cycles={len(tfr_cycles)}  n_brain_ics={n_brain}")

    return {
        "subject":            subject,
        "beta_double_stance": float(beta_double_stance),
        "beta_swing":         float(beta_swing),
        "beta_diff":          float(beta_double_stance - beta_swing),
        "n_cycles":           len(tfr_cycles),
        "n_brain_ics":        int(n_brain),
        **decisions,
    }
