"""
ICA utilities: fitting, ICLabel classification, and QC plotting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_icalabel import label_components
from mne_icalabel.iclabel import iclabel_label_components

# Column order of the raw ICLabel probability matrix
# (mne_icalabel.label_components.ICALABEL_METHODS_NUMERICAL_TO_STRING["iclabel"]).
ICLABEL_CLASSES = [
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
]


def run_ica(
    epochs: mne.Epochs,
    n_components: float | int = 0.99,
    method: str = "infomax",
    fit_params: dict | None = None,
    decim: int = 2,
    random_state: int = 42,
) -> mne.preprocessing.ICA:
    """Fit Extended Infomax ICA on clean epochs and return the fitted ICA object."""

    rank = mne.compute_rank(epochs, rank="info")
    print(f"  Data rank        : {rank}")

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(epochs, decim=decim)
    print(f"  ICA components   : {ica.n_components_}")
    return ica


def label_and_mark_ica(
    ica: mne.preprocessing.ICA,
    epochs: mne.Epochs,
    brain_thresh: float = 0.7,
) -> dict:
    """Run ICLabel on epochs, mark components below brain_thresh for exclusion, return updated ICA."""

    result = label_components(epochs, ica, method="iclabel")
    labels = result["labels"]
    probs  = result["y_pred_proba"]

    brain_ics, exclude_ics = [], []

    print("\n  ICLabel classification:")
    for i, (label, prob_vec) in enumerate(zip(labels, probs)):
        prob   = float(prob_vec.max())
        keep   = label == "brain" and prob >= brain_thresh
        marker = "+" if keep else "-"
        print(f"    {marker}  IC{i:03d}  {label:<20}  p={prob:.2f}")
        (brain_ics if keep else exclude_ics).append(i)

    ica.exclude = exclude_ics
    print(f"\n  Keeping  : {len(brain_ics)} brain ICs  {brain_ics}")
    print(f"  Excluding: {len(exclude_ics)} artefact ICs")

    return {
        "labels":      labels,
        "probs":       probs,
        "brain_ics":   brain_ics,
        "exclude_ics": exclude_ics,
    }


def iclabel_probabilities(
    ica: mne.preprocessing.ICA,
    epochs: mne.Epochs,
) -> tuple[np.ndarray, list[str]]:
    """
    Run ICLabel and return the full per-class probability matrix.

    Unlike label_and_mark_ica() / mne_icalabel.label_components(), this
    keeps every class's probability (not just the argmax), so a caller
    can apply more than one exclusion rule to the same ICLabel run
    without recomputing it (see select_ics_by_rule()).

    Parameters
    ----------
    ica    : fitted mne.preprocessing.ICA
    epochs : mne.Epochs used to fit `ica` (ICLabel needs these for features)

    Returns
    -------
    probs  : ndarray, shape (n_components, 7)
        Columns ordered per ICLABEL_CLASSES: brain, muscle artifact,
        eye blink, heart beat, line noise, channel noise, other.
    labels : list of str, length n_components
        Argmax class label per component (for logging only).
    """
    probs = iclabel_label_components(epochs, ica)
    labels = [ICLABEL_CLASSES[i] for i in probs.argmax(axis=1)]
    return probs, labels


def select_ics_by_rule(probs: np.ndarray, rule: str) -> list[int]:
    """
    Decide which ICA components to exclude from an existing ICLabel
    probability matrix, without refitting ICA or ICLabel.

    Parameters
    ----------
    probs : ndarray, shape (n_components, 7)
        From iclabel_probabilities(), columns ordered per ICLABEL_CLASSES.
    rule  : {"conservative", "balanced", "liberal"}
        conservative -> keep only ICs with P(brain) > 0.9
        balanced     -> keep only ICs with P(brain) > 0.7
        liberal      -> reject only ICs with P(muscle) > 0.9 or P(eye) > 0.9

    Returns
    -------
    exclude_ics : list of int
        Component indices to exclude (assign to ica.exclude).
    """
    p_brain, p_muscle, p_eye = probs[:, 0], probs[:, 1], probs[:, 2]
    if rule == "conservative":
        keep = p_brain > 0.9
    elif rule == "balanced":
        keep = p_brain > 0.7
    elif rule == "liberal":
        keep = ~((p_muscle > 0.9) | (p_eye > 0.9))
    else:
        raise ValueError(f"Unknown iclabel_rule: {rule!r}")
    return [i for i, k in enumerate(keep) if not k]


def save_ica_component_plots(
    ica: mne.preprocessing.ICA,
    output_dir: Path,
    subject: str,
) -> None:
    """Save topography plots for all ICA components to the output directory."""

    figs = ica.plot_components(show=False)
    if not isinstance(figs, list):
        figs = [figs]

    for k, fig in enumerate(figs):
        out = output_dir / f"sub-{subject}_ica_topos_{k:02d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ICA topos        -> {out.name}")
