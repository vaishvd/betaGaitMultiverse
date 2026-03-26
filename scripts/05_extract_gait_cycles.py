import mne
import numpy as np
from src.config import DIR_ICA, DIR_RAWDATA, DIR_GAIT
from src.gait_cycles import (
    load_events,
    filter_condition,
    extract_rhs_cycles,
    get_toe_off_onsets,
    extract_cycle_simple,
    extract_cycle_twophase,
    print_cycle_qc,
)

SUBJECTS = ["S18"]

N_TOTAL  = 200  # time points per cycle (simple normalization)
N_STANCE = 120  # time points for stance phase (~60% of cycle)
N_SWING  =  80  # time points for swing phase  (~40% of cycle)

for sub in SUBJECTS:
    print(f"\nProcessing {sub}")

    raw = mne.io.read_raw_fif(DIR_ICA / f"sub-{sub}_desc-clean_raw.fif", preload=True)
    sfreq = raw.info["sfreq"]

    events_df = load_events(DIR_RAWDATA / f"sub-{sub}/eeg/sub-{sub}_task-task_events.tsv")
    events_df = filter_condition(events_df, "B3", "End B3")
    cycles    = extract_rhs_cycles(events_df)   # list of (start_sec, end_sec)
    rto       = get_toe_off_onsets(events_df)   # array of toe-off times, or None

    all_cycles_simple   = []
    all_cycles_twophase = []
    cycle_durations     = []

    for start_sec, end_sec in cycles:

        if int(np.round(end_sec * sfreq)) > raw.n_times:
            print(f"  Skipping cycle at {start_sec:.2f}s — beyond recording end")
            continue

        cycle_durations.append(end_sec - start_sec)

        # Simple whole-cycle normalization
        all_cycles_simple.append(
            extract_cycle_simple(raw, start_sec, end_sec, sfreq, N_TOTAL)
        )

        # Two-phase normalization (only if toe-off events are available)
        if rto is not None:
            toe_offs_in_cycle = rto[(rto >= start_sec) & (rto < end_sec)]
            if len(toe_offs_in_cycle) == 1:
                result = extract_cycle_twophase(
                    raw, start_sec, end_sec, toe_offs_in_cycle[0],
                    sfreq, N_STANCE, N_SWING
                )
                if result is not None:
                    all_cycles_twophase.append(result)

    # Save and report
    all_cycles_simple = np.stack(all_cycles_simple)  # (n_cycles, n_channels, N_TOTAL)
    print(f"  Extracted {all_cycles_simple.shape[0]} cycles, shape {all_cycles_simple.shape[1:]}")
    print_cycle_qc(cycle_durations)

    np.save(DIR_GAIT / f"sub-{sub}_gait_cycles.npy", all_cycles_simple)
    print(f"  Saved simple-normalized cycles → sub-{sub}_gait_cycles.npy")

    if all_cycles_twophase:
        all_cycles_twophase = np.stack(all_cycles_twophase)  # (n_cycles, n_channels, N_STANCE + N_SWING)
        np.save(DIR_GAIT / f"sub-{sub}_gait_cycles_twophase.npy", all_cycles_twophase)
        print(f"  Saved two-phase-normalized cycles → sub-{sub}_gait_cycles_twophase.npy")