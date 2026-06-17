"""
Gaussian spatial filter for EEG beta ERSP analysis.

Computes distance-weighted channel weights centered on a reference
electrode. Weights decay as a Gaussian function of 3D Euclidean
distance from the reference channel. This is used to compute a
spatially smoothed sensorimotor ROI rather than selecting a single
channel or averaging a fixed channel list.

The weight topography can be plotted as a Methods figure to show
which channels contribute to the analysis.

Reference: Seeber et al. 2015 J Neurosci use a similar
distance-weighted sensorimotor ROI for beta ERSP analysis.
"""

import numpy as np
import mne
from pathlib import Path


def gaussian_roi_weights(
    info: mne.Info,
    center_ch: str = "Cz",
    sigma_mm: float = 40.0,
) -> np.ndarray:
    """
    Compute Gaussian spatial weights centered on a reference channel.

    Weights are proportional to exp(-d^2 / (2*sigma^2)) where d is
    the 3D Euclidean distance from the center channel in millimetres.
    Weights are normalised to sum to 1.0.

    Channels with missing or zero electrode positions receive weight 0.

    Parameters
    ----------
    info      : mne.Info containing electrode positions
    center_ch : name of the reference channel (default "Cz")
    sigma_mm  : Gaussian spread in millimetres (default 40 mm,
                approximately 2-3 electrode spacings). Smaller values
                produce a tighter sensorimotor ROI; larger values
                approach a flat average.

    Returns
    -------
    weights : np.ndarray shape (n_ch,)
        Normalised Gaussian weights. Sums to 1.0.

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

    # Zero out channels with missing positions
    has_pos = np.array([
        np.any(positions[i] != 0) for i in range(len(info.ch_names))
    ])
    dists[~has_pos] = np.inf

    # Gaussian weights
    weights = np.exp(-dists**2 / (2.0 * sigma_mm**2))
    weights[~has_pos] = 0.0

    # Normalise
    total = weights.sum()
    if total == 0:
        raise ValueError("All channel weights are zero — check positions")
    weights /= total

    return weights


def apply_gaussian_roi(
    ersp: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Apply Gaussian spatial weights to an ERSP array.

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
        Normalised weights from gaussian_roi_weights().

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
    center_ch: str = "Cz",
    sigma_mm: float = 40.0,
) -> None:
    """
    Save a topography plot of the Gaussian ROI weights.

    Parameters
    ----------
    weights   : weight vector from gaussian_roi_weights()
    info      : mne.Info with electrode positions
    subject   : subject id for title
    out_path  : output PNG path
    center_ch : reference channel name
    sigma_mm  : Gaussian sigma used (for title label)
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
    )
    ax.set_title(
        f"sub-{subject}  Gaussian ROI weights\n"
        f"center={center_ch}  sigma={sigma_mm}mm",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
