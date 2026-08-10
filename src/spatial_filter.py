"""
Linear spatial filter for EEG beta ERSP analysis.

Computes distance-weighted channel weights centered on a reference
electrode. Weights decay linearly with 3D Euclidean distance from the
reference channel: w_i = 1 - (d_i / d_max), where d_max is the maximum
distance across all electrodes with valid positions. This defines a
sensorimotor ROI that weights near-center channels more strongly.

The weight topography can be plotted as a Methods figure to show
which channels contribute to the analysis.

References:
  Petersen et al. 2012 J Physiol 590:2443 — beta ERS at heel contact
  Bulea et al. 2015 Front Hum Neurosci 9:247 — beta ERD during swing
  Seeber et al. 2015 Front Hum Neurosci 9:1 — gait phase beta dynamics
"""

import numpy as np
import mne
from pathlib import Path

from src.config import ROI_CENTER_CH

# mne.viz.plot_topomap's projection sphere, made explicit rather than
# left to sphere='auto'/'eeglab'. This montage's fiducials (LPA/RPA/
# Nasion) sit exactly at z=0 in this coordinate frame, i.e. (0, 0, 0) IS
# MNE/Neuromag's own head-frame origin here -- not a custom value. But
# the electrode cloud is a realistic (non-spherical) head shape: ear-
# level channels sit at radius ~0.09-0.093 from that origin while vertex
# channels (Cz, CPz, Fz...) sit out at ~0.14. Least-squares sphere fits
# ('auto') and horizon fits ('eeglab') both try to fit ONE sphere to that
# whole elongated cloud and get dragged upward (center z~+0.04) by the
# many far-out superior channels -- confirmed by comparing sensor-only
# renders under 'auto'/'eeglab' (electrodes bunch inward, well short of
# the head outline) vs this origin sphere (electrodes fill the outline
# uniformly, matching known landmarks: Cz central, Fz anterior, Oz
# posterior, T7/T8 lateral). 0.095 m is the nominal adult head radius
# already used when these positions were built (HEAD_RADIUS_M in
# src/pipeline_steps.py). This is a display/projection choice only; it
# does not touch channel positions, ROI weights (linear_roi_weights
# below uses raw 3D distances, no sphere), or any ERSP/statistical
# computation.
TOPOMAP_SPHERE = (0.0, 0.0, 0.0, 0.095)


def linear_roi_weights(
    info: mne.Info,
    center_ch: str = ROI_CENTER_CH,
) -> np.ndarray:
    """
    Compute linear distance weights centered on a reference channel.

    w_i = 1 - (d_i / d_max), where d_i is the 3D Euclidean distance from
    each electrode to center_ch and d_max is the maximum distance across
    all electrodes with valid positions. Weights are normalised to sum to 1.0.

    Channels with missing or zero electrode positions receive weight 0.

    Parameters
    ----------
    info      : mne.Info containing electrode positions
    center_ch : name of the reference channel (default "Cz")

    Returns
    -------
    weights : np.ndarray shape (n_ch,)
        Normalised linear weights. Sums to 1.0.

    Raises
    ------
    ValueError if center_ch is not in info.ch_names or has no position.
    """
    if center_ch not in info.ch_names:
        raise ValueError(f"Center channel '{center_ch}' not in info")

    # Extract 3D positions in metres, convert to mm
    positions = np.array([
        info["chs"][i]["loc"][:3] * 1000.0
        for i in range(len(info.ch_names))
    ])  # (n_ch, 3) in mm

    center_idx = info.ch_names.index(center_ch)
    center_pos = positions[center_idx]

    if not np.any(center_pos != 0):
        raise ValueError(
            f"Center channel '{center_ch}' has no electrode position"
        )

    # Euclidean distance from center in mm
    dists = np.linalg.norm(positions - center_pos, axis=1)  # (n_ch,)

    # Identify channels with valid positions
    has_pos = np.array([
        np.any(positions[i] != 0) for i in range(len(info.ch_names))
    ])

    valid_dists = dists[has_pos]
    d_max = valid_dists.max()
    if d_max == 0:
        raise ValueError("All electrodes are at the same position — check positions")

    # Linear weights: w_i = 1 - (d_i / d_max), clamped to [0, 1]
    weights = np.zeros(len(info.ch_names))
    weights[has_pos] = np.clip(1.0 - (valid_dists / d_max), 0.0, 1.0)

    # Normalise
    total = weights.sum()
    if total == 0:
        raise ValueError("All channel weights are zero — check positions")
    weights /= total

    return weights


def apply_linear_roi(
    ersp: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Apply linear ROI weights to an ERSP array.

    Computes the weighted sum across channels, collapsing the channel
    dimension.

    Parameters
    ----------
    ersp    : np.ndarray
        ERSP array. Channel axis must be axis 0.
        Accepted shapes:
          (n_ch, n_freqs, n_times)  → returns (n_freqs, n_times)
          (n_ch, n_freqs)           → returns (n_freqs,)
    weights : np.ndarray shape (n_ch,)
        Normalised weights from linear_roi_weights().

    Returns
    -------
    np.ndarray with channel dimension collapsed.
    """
    if ersp.shape[0] != weights.shape[0]:
        raise ValueError(
            f"Channel count mismatch: ersp has {ersp.shape[0]} channels "
            f"but weights has {weights.shape[0]}"
        )
    # Reshape weights for broadcasting: (n_ch,) → (n_ch, 1, 1) or (n_ch, 1)
    w = weights.reshape((-1,) + (1,) * (ersp.ndim - 1))
    return (ersp * w).sum(axis=0)


def plot_weight_topography(
    weights: np.ndarray,
    info: mne.Info,
    subject: str,
    out_path: Path,
    center_ch: str = ROI_CENTER_CH,
) -> None:
    """
    Save a topography plot of the linear ROI weights.

    Parameters
    ----------
    weights   : weight vector from linear_roi_weights()
    info      : mne.Info with electrode positions
    subject   : subject id for title
    out_path  : output PNG path
    center_ch : reference channel name
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    mne.viz.plot_topomap(
        weights,
        info,
        axes=ax,
        show=False,
        cmap="Reds",
        vlim=(0, weights.max()),
        contours=4,
        sphere=TOPOMAP_SPHERE,
    )
    ax.set_title(
        f"sub-{subject}  Linear ROI weights\n"
        f"center={center_ch}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
