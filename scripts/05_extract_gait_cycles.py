import mne
import numpy as np
from src.config import DIR_ICA, DIR_RAWDATA, DIR_GAIT
from src.gait_cycles import (
    load_events,
    filter_condition,
    rhs_cycles,
    compute_durations,
    plot_duration_distribution
)

SUBJECTS = ["S18"]

for sub in SUBJECTS:
    print(f"\n{sub} — Extracting gait cycles")

    # Load raw EEG
    raw = mne.io.read_raw_fif(
        DIR_ICA / f"sub-{sub}_desc-clean_raw.fif",
        preload=False
    )

    # Load events 
    events = load_events(
        DIR_RAWDATA / f"sub-{sub}/eeg/sub-{sub}_task-task_events.tsv"
    )

    # align to raw time axis
    events["onset"] -= raw.first_time

    # Keep only walking block
    events = filter_condition(events, "B3", "End B3")

    # Extract gait cycles (RHS to RHS)
    cycles = rhs_cycles(events)
    print(f"  Total cycles: {len(cycles)}")

    # Compute durations 
    durations = compute_durations(cycles)

    print(f"  Duration: {durations.mean():.2f} ± {durations.std():.2f} s")
    print(f"  Range   : {durations.min():.2f} – {durations.max():.2f} s")

    # Save cycles
    np.save(DIR_GAIT / f"sub-{sub}_cycles.npy", cycles)
    np.save(DIR_GAIT / f"sub-{sub}_durations.npy", durations)

    # Plot distribution
    out_plot = DIR_GAIT / f"sub-{sub}_cycle_duration.png"
    plot_duration_distribution(durations, out_plot)

    print(f"  Saved cycles + duration plot")