import numpy as np
import mne
import matplotlib.pyplot as plt
from src.config import DIR_ERSP, DIR_ICA, DIR_PLOTS, DIR_RAWDATA
from src.gait_cycles import load_events, filter_condition, rhs_cycles, compute_event_means

SUBJECTS = ["S18"]
CHANNELS = ["A1"]
FREQS    = np.arange(13, 31, dtype=float)

for sub in SUBJECTS:
    print(f"\n {sub} — Plotting beta ERSP ...")

    # Load ERSP and select channels
    ersp = np.load(DIR_ERSP / f"sub-{sub}_ersp_beta.npy")    # (ch, freqs, n_points)
    raw  = mne.io.read_raw_fif(DIR_ICA / f"sub-{sub}_desc-clean_raw.fif", preload=False)

    ch_idx   = [raw.ch_names.index(ch) for ch in CHANNELS]
    ersp_sel = ersp[ch_idx].mean(axis=0)                     # (freqs, n_points)

    # Recompute gait event positions in cycle space
    events = load_events(DIR_RAWDATA / f"sub-{sub}/eeg/sub-{sub}_task-task_events.tsv")
    events["onset"] -= raw.first_time
    events = filter_condition(events, "B3", "End B3")
    cycles = rhs_cycles(events)

    event_means = compute_event_means(events, cycles)
    event_means["RHS"] = 0.0    # RHS defines cycle start by definition
    print("  Event means (% cycle):", event_means)

    # Plot 
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
    plt.colorbar(im, ax=ax, label="Power change (dB)")

    for ev, xval in event_means.items():
        if xval is None:
            continue
        ax.axvline(xval, ls="--", color="black", lw=1)
        ax.text(xval, FREQS[0] - 0.8, ev, ha="center", fontsize=8)

    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{sub} — Beta ERSP ({', '.join(CHANNELS)})")
    fig.tight_layout()

    out = DIR_PLOTS / f"sub-{sub}_beta_ersp.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out.name}")