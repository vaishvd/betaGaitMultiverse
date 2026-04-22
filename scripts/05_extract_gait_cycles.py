import mne
import numpy as np
from src.config import DIR_ICA, DIR_RAWDATA, DIR_GAIT
from src.gait_cycles import (
    load_events,
    filter_condition,
    rhs_cycles,
    extract_cycles,
    plot_cycle_diagnostics
)

SUBJECTS = ["S18"]
N_POINTS = 512  # time points for time-normalisation (512 = 2 s at 256 Hz) 
MIN_DUR  = 0.5   
MAX_DUR  = 2.5   


for sub in SUBJECTS:
    print(f"\n {sub} - Extracting gait cycles")

    # Load raw data (after ICA cleaning)
    raw = mne.io.read_raw_fif(DIR_ICA / f"sub-{sub}_desc-clean_raw.fif", preload=True)

    # Load events; subtract raw.first_time to align onsets to the EEG time axis
    events = load_events(DIR_RAWDATA / f"sub-{sub}/eeg/sub-{sub}_task-task_events.tsv")
    events["onset"] -= raw.first_time
    events = filter_condition(events, "B3", "End B3")

    cycles = rhs_cycles(events)
    print(f"  RHS cycles : {len(cycles)}")

    # Diagnostic: raw alignment before any rejection
    plot_cycle_diagnostics(cycles, raw, DIR_GAIT / f"sub-{sub}_cycle_diagnostics.png")

    # Extract, reject, and time-normalise cycles
    all_cycles, durations = extract_cycles(
        raw, cycles, raw.info["sfreq"], MIN_DUR, MAX_DUR, N_POINTS, k=3.0,
    )

    if all_cycles is None:
        print("  No cycles extracted — skipping.")
        continue

    print(f"  Cycles kept : {len(all_cycles)}")
    print(f"  Duration    : {durations.mean():.2f} ± {durations.std():.2f} s")

    # Save cycles and average
    np.save(DIR_GAIT / f"sub-{sub}_gait_cycles.npy", all_cycles)   # (n, ch, time)
    np.save(DIR_GAIT / f"sub-{sub}_gait_avg.npy",    all_cycles.mean(axis=0))  # (ch, time)
    print(f"  Saved : {all_cycles.shape}")