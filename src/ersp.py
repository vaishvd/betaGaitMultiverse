"""
Shared event-anchored gait-cycle warping and beta double-stance-vs-swing
phase-contrast logic.

Single source of truth for the warp/phase-split/beta-reduction math used
identically by the canonical pipeline (prepana05_gaitcycles2tfr.py,
prepana07_betaphase_stats.py) and the multiverse pipeline
(multiverse_pipeline.py), so every universe computes the same contrast
definition the canonical pipeline validated.
"""

import json

import mne
import numpy as np

from src.spatial_filter import apply_linear_roi

BETA_FMIN, BETA_FMAX = 13.0, 30.0


def load_group_anchors(ersp_dir):
    """
    Load the group-median gait-event anchors written once by
    prepana05_gaitcycles2tfr.py (pooled across all canonical subjects'
    kept cycles).

    Parameters
    ----------
    ersp_dir : Path
        Dataset ERSP directory (``dirs["ersp"]``) containing
        ``group_gait_event_anchors.json``.

    Returns
    -------
    (A_lto, A_lhs, A_rto) : tuple of float
        Percent of gait cycle (0-100) for each event anchor.
    """
    with open(ersp_dir / "group_gait_event_anchors.json") as f:
        anchors = json.load(f)
    return anchors["A_lto_pct"], anchors["A_lhs_pct"], anchors["A_rto_pct"]


def warp_cycle_to_grid(power, lto_idx, lhs_idx, rto_idx, anchors, n_points=101):
    """
    Piecewise-linear (4-segment) time-warp of one cycle's power time
    series onto the common 0-100% grid, pinning this cycle's own
    LTO/LHS/RTO sample positions to the group-median anchor percentages.

    Parameters
    ----------
    power : ndarray, shape (n_ch, n_freqs, n_t)
        One cycle's TFR power, already edge-cropped.
    lto_idx, lhs_idx, rto_idx : int
        This cycle's own event sample positions within `power`'s time
        axis (i.e. already adjusted for the same edge-crop as `power`).
    anchors : tuple of float
        (A_lto, A_lhs, A_rto) percent-of-cycle anchors, shared across
        every cycle/subject/universe (see load_group_anchors()).
    n_points : int, optional
        Output grid size (default 101, 0-100% in 1% steps).

    Returns
    -------
    ndarray, shape (n_ch, n_freqs, n_points)

    Raises
    ------
    ValueError
        If ``0 < lto_idx < lhs_idx < rto_idx < n_t - 1`` does not hold
        -- the caller should treat this cycle as invalid and skip it.
    """
    A_lto, A_lhs, A_rto = anchors
    n_t = power.shape[-1]
    if not (0 < lto_idx < lhs_idx < rto_idx < n_t - 1):
        raise ValueError(
            f"Invalid event order for warp: lto={lto_idx} lhs={lhs_idx} "
            f"rto={rto_idx} n_t={n_t}"
        )
    old_bp = np.array([0, lto_idx, lhs_idx, rto_idx, n_t - 1], dtype=float)
    pct_bp = np.array([0.0, A_lto, A_lhs, A_rto, 100.0])
    x_old  = np.interp(np.arange(n_t), old_bp, pct_bp)   # warped time axis, in %
    x_new  = np.linspace(0, 100, n_points)
    return np.stack([
        np.stack([np.interp(x_new, x_old, power[ch, f])
                  for f in range(power.shape[1])])
        for ch in range(power.shape[0])
    ])   # (n_ch, n_freqs, n_points)


def phase_split_indices(anchors, n_points=101):
    """
    Fixed double-stance / swing index ranges on the common warped grid.

    Double support occurs twice per gait cycle (initial DS: 0 -> LTO;
    terminal DS: LHS -> RTO); swing likewise occurs twice (LTO -> LHS,
    RTO -> 100%). Valid once every cycle has been warped with
    warp_cycle_to_grid() using the same anchors, since every cycle's
    LTO/LHS/RTO then lands on the same grid indices.

    Parameters
    ----------
    anchors : tuple of float
        (A_lto, A_lhs, A_rto) percent-of-cycle anchors.
    n_points : int, optional
        Grid size (default 101).

    Returns
    -------
    (double_stance_idx, swing_idx) : tuple of ndarray
        Integer grid indices for each phase.
    """
    A_lto, A_lhs, A_rto = anchors
    idx_lto = int(round(A_lto / 100 * (n_points - 1)))
    idx_lhs = int(round(A_lhs / 100 * (n_points - 1)))
    idx_rto = int(round(A_rto / 100 * (n_points - 1)))

    double_stance_idx = np.concatenate([
        np.arange(0, idx_lto),
        np.arange(idx_lhs, idx_rto),
    ])
    swing_idx = np.concatenate([
        np.arange(idx_lto, idx_lhs),
        np.arange(idx_rto, n_points),
    ])
    return double_stance_idx, swing_idx


def compute_standing_baseline(
    raw_stand, freqs, n_cycles, edge_crop=0.05, amp_thresh=350e-6, epoch_dur=2.0,
):
    """
    Compute standing-baseline Morlet power for ERSP normalization.

    Builds fixed-length epochs from a quiet-standing raw segment, drops
    epochs exceeding `amp_thresh`, computes per-epoch Morlet power,
    edge-crops each epoch's power in time, and averages over epochs
    and time.

    Parameters
    ----------
    raw_stand  : mne.io.BaseRaw
        The quiet-standing segment (already cropped to the STAND
        annotation, with any trailing boundary artefact trimmed).
    freqs      : ndarray, shape (n_freqs,)
        Morlet wavelet frequencies (Hz).
    n_cycles   : ndarray, shape (n_freqs,)
        Morlet wavelet cycle count per frequency.
    edge_crop  : float, optional
        Fraction of the epoch trimmed at each edge post-TFR (default 0.05).
    amp_thresh : float, optional
        Peak absolute amplitude (V) above which an epoch is rejected
        (default 350e-6).
    epoch_dur  : float, optional
        Fixed epoch duration in seconds (default 2.0).

    Returns
    -------
    baseline_power : ndarray, shape (n_ch, n_freqs)

    Raises
    ------
    RuntimeError
        If every standing epoch is rejected.
    """
    events = mne.make_fixed_length_events(raw_stand, duration=epoch_dur)
    stand_epochs = mne.Epochs(
        raw_stand, events,
        tmin=0, tmax=epoch_dur,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )
    stand_epochs.drop_bad(reject=dict(eeg=amp_thresh))

    tfr_list = []
    for epoch in stand_epochs.get_data():
        if np.max(np.abs(epoch)) > amp_thresh:
            continue
        tfr = mne.time_frequency.tfr_array_morlet(
            epoch[np.newaxis],
            sfreq=raw_stand.info["sfreq"],
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            zero_mean=True,
            verbose=False,
        )[0]
        crop = int(edge_crop * tfr.shape[-1])
        tfr_list.append(tfr[..., crop:-crop] if crop else tfr)

    if not tfr_list:
        raise RuntimeError("All standing epochs rejected -- cannot compute baseline")

    stack = np.stack(tfr_list)   # (n_kept, n_ch, n_freqs, n_time_cropped)
    return stack.mean(axis=(0, 3))   # (n_ch, n_freqs)


def apply_gpm_normalization(ersp):
    """
    Gait-phase-mean (GPM) re-normalization of an already baseline-
    normalized ERSP array: express each (channel, frequency) row's ERSP
    as dB relative to that row's OWN mean across the whole gait cycle,
    instead of dB relative to the standing/rest baseline.

        ERSP_gpm(..., t) = ERSP(..., t) - mean_t[ERSP(..., t)]

    Operates on the last axis (the gait-cycle-% axis, e.g. the 101-point
    grid from warp_cycle_to_grid); works on any leading shape --
    (n_ch, n_freqs, n_points), (n_freqs, n_points), etc.

    This is a pure post-hoc transform of the standing-baselined ERSP
    dB array -- it does not touch the underlying TFR power, the
    standing baseline, or any raw/ICA data. Because it subtracts a
    per-(channel, frequency) CONSTANT (uniform across t) from every
    gait-cycle point, it cancels EXACTLY in any linear contrast between
    two subsets of t -- in particular the double-stance-mean-minus-
    swing-mean beta contrast used by prepana07/multiverse_pipeline is
    mathematically invariant to this transform, and the group paired
    t-test is identical whether GPM or standing normalization is used
    (see prepana07_betaphase_stats.py's cross-mode verification).
    """
    return ersp - ersp.mean(axis=-1, keepdims=True)


def beta_roi_scalar(ersp_map, weights, freqs, fmin=BETA_FMIN, fmax=BETA_FMAX):
    """
    Reduce a (n_ch, n_freqs) ERSP map to a single beta-band scalar using
    linear Cz-ROI channel weights.

    Parameters
    ----------
    ersp_map : ndarray, shape (n_ch, n_freqs)
    weights  : ndarray, shape (n_ch,)
        From src.spatial_filter.linear_roi_weights().
    freqs    : ndarray, shape (n_freqs,)
        Frequency axis matching `ersp_map`'s second dimension.
    fmin, fmax : float, optional
        Beta band bounds in Hz (default 13-30).

    Returns
    -------
    float
    """
    roi  = apply_linear_roi(ersp_map, weights)   # (n_freqs,)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(roi[mask].mean())
