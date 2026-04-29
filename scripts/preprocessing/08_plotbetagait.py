import numpy as np
import mne
import matplotlib.pyplot as plt

from src.paths import get_subject_paths, get_dataset_dirs
from src.config import DATASETS, DIR_RESULTS

from src.gait_cycles import (
    load_events,
    filter_condition,
    rhs_cycles,
    compute_event_means,
)

DATASET = "splitbelt"
SUBJECTS = ["S18"]

CHANNELS = ["A1"]
FREQS = np.arange(13, 31, dtype=float)

cfg  = DATASETS[DATASET]
dirs = get_dataset_dirs(DATASET)

results_dir = DIR_RESULTS / DATASET
plot_dir = results_dir / "plots"

for sub in SUBJECTS:
    print(f"\n{sub} — Plotting beta ERSP")

    paths = get_subject_paths(DATASET, sub)

    ersp = np.load(dirs["ersp"] / f"sub-{sub}_ersp_beta.npy")
    raw  = mne.io.read_raw_fif(paths["ica"], preload=False)

    ch_idx = [raw.ch_names.index(ch) for ch in CHANNELS]
    ersp_sel = ersp[ch_idx].mean(axis=0)

    events = load_events(paths["events"])
    events = filter_condition(
        events,
        cfg["condition_start"],
        cfg["condition_end"],
    )

    cycles = rhs_cycles(events)
    event_means = compute_event_means(events, cycles)
    event_means["RHS"] = 0.0

    vmax = np.max(np.abs(ersp_sel))

    fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(
        ersp_sel,
        aspect="auto",
        origin="lower",
        extent=[0, 100, FREQS[0], FREQS[-1]],
        cmap="turbo",
        vmin=-vmax,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax)

    for ev, x in event_means.items():
        if x is None:
            continue
        ax.axvline(x, ls="--", color="black", lw=1)

    out = plot_dir / f"sub-{sub}_beta_ersp.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)