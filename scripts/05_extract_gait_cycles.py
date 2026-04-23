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
MIN_DUR  = 0.5   
MAX_DUR  = 2.5   

for sub in SUBJECTS:
    print(f"\n {sub} - Extracting gait cycles")

    # Load raw data (after ICA cleaning)
    raw = mne.io.read_raw_fif(DIR_ICA / f"sub-{sub}_desc-clean_raw.fif", preload=True)
    sfreq = raw.info["sfreq"]

    # Load events; subtract raw.first_time to align onsets to the EEG time axis
    events = load_events(DIR_RAWDATA / f"sub-{sub}/eeg/sub-{sub}_task-task_events.tsv")
    events["onset"] -= raw.first_time
    events = filter_condition(events, "B3", "End B3")

    cycles = rhs_cycles(events)
    print(f"  RHS cycles : {len(cycles)}")

    # Diagnostic plot: raw alignment before any rejection
    plot_cycle_diagnostics(cycles, raw, DIR_GAIT / f"sub-{sub}_cycle_diagnostics.png")
 
    segments, durations = extract_cycles(raw, cycles, sfreq, MIN_DUR, MAX_DUR, k=3.0)
 
    if segments is None:
        print("  No cycles survived — skipping.")
        continue
 
    print(f"  Cycles kept : {len(segments)}")
    print(f"  Duration    : {durations.mean():.2f} ± {durations.std():.2f} s")
 
    # Segments are variable-length — pre-allocate object array to prevent
    # NumPy from broadcasting into a 3D array when all strides share the same length
    raw_arr = np.empty(len(segments), dtype=object)
    for i, seg in enumerate(segments):
        raw_arr[i] = seg
 
    np.save(DIR_GAIT / f"sub-{sub}_gait_segments.npy",  raw_arr,   allow_pickle=True)
    np.save(DIR_GAIT / f"sub-{sub}_gait_durations.npy", durations)
    np.save(DIR_GAIT / f"sub-{sub}_gait_sfreq.npy",     sfreq)
    print(f"  Saved : {len(segments)} segments, sfreq={sfreq} Hz")