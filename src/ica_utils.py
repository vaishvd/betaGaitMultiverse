"""
ICA utilities: fitting, ICLabel classification, and QC plotting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_icalabel import label_components


def load_epochs_ica(subject: str, epoch_file: Path) -> mne.Epochs:
    """Load pre-ICA clean epochs and pick EEG channels."""
    epochs = mne.read_epochs(epoch_file, preload=True)
    epochs.pick("eeg")
    return epochs


def load_raw_for_ica(subject: str, sigclean_dir: Path) -> mne.io.BaseRaw:
    """Load the sigclean raw FIF for ICA application."""
    fname = sigclean_dir / f"sub-{subject}_clean_raw.fif"
    return mne.io.read_raw_fif(fname, preload=True)


def run_ica(
    epochs: mne.Epochs,
    n_components: float | int = 0.99,
    method: str = "infomax",
    fit_params: dict | None = None,
    decim: int = 2,
    random_state: int = 42,
) -> mne.preprocessing.ICA:
    """
    Fit ICA on clean epochs.

    Parameters
    ----------
    n_components  : float (variance threshold) or int (explicit count)
    fit_params    : extra kwargs forwarded to mne.preprocessing.ICA (e.g. dict(extended=True))
    decim         : decimation factor during fitting (speeds up computation)

    Returns
    -------
    Fitted ICA object (no components excluded yet).
    """
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
    """
    Run ICLabel and mark artefact components for exclusion on the ICA object.

    A component is kept as brain if:
      - ICLabel classifies it as 'brain', AND
      - the classification probability is >= brain_thresh

    All other components are added to ica.exclude.

    Parameters
    ----------
    brain_thresh : probability threshold for accepting a 'brain' label

    Returns
    -------
    dict with keys:
        labels      : list of string labels per component
        probs       : array of per-class probabilities
        brain_ics   : indices of kept (brain) components
        exclude_ics : indices of excluded (artefact) components
    """
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


def save_ica_component_plots(
    ica: mne.preprocessing.ICA,
    output_dir: Path,
    subject: str,
) -> None:
    """
    Save ICA component topography figures to output_dir.
    One file per figure page (MNE splits into pages for large component sets).
    """
    figs = ica.plot_components(show=False)
    if not isinstance(figs, list):
        figs = [figs]

    for k, fig in enumerate(figs):
        out = output_dir / f"sub-{subject}_ica_topos_{k:02d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ICA topos        -> {out.name}")


def apply_iclabel(ica: mne.preprocessing.ICA, epochs: mne.Epochs):
    """
    Legacy helper: classify with ICLabel (no threshold) and apply ICA to epochs.
    Prefer label_and_mark_ica() for pipeline use — it supports a probability
    threshold and does not apply ICA (application belongs in ana03).
    """
    result = label_components(epochs, ica, method="iclabel")

    ica.exclude = [
        i for i, label in enumerate(result["labels"])
        if label != "brain"
    ]
    print(f"Excluding {len(ica.exclude)} components: {ica.exclude}")

    epochs_clean = ica.apply(epochs.copy())
    return epochs_clean, ica
