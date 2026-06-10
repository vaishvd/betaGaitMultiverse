"""
ICA utilities: fitting, ICLabel classification, and QC plotting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_icalabel import label_components


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
