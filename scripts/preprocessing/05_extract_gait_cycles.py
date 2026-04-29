import mne
import numpy as np

from src.paths import get_subject_paths, get_dataset_dirs
from src.config import DATASETS

from src.gait_cycles import (
    load_events,
    filter_condition,
    rhs_cycles,
    compute_durations,
)

DATASET = "splitbelt"
SUBJECTS = ["S18"]

cfg  = DATASETS[DATASET]
dirs = get_dataset_dirs(DATASET)

for sub in SUBJECTS:
    print(f"\n{sub} — Extracting gait cycles")

    paths = get_subject_paths(DATASET, sub)

    raw = mne.io.read_raw_fif(paths["ica"], preload=False)

    events = load_events(paths["events"])
    events = filter_condition(
        events,
        cfg["condition_start"],
        cfg["condition_end"],
    )

    cycles = rhs_cycles(events)
    durations = compute_durations(cycles)

    np.save(dirs["gait"] / f"sub-{sub}_cycles.npy", cycles)
    np.save(dirs["gait"] / f"sub-{sub}_durations.npy", durations)

    # also save sfreq 
    np.save(dirs["gait"] / f"sub-{sub}_sfreq.npy", raw.info["sfreq"])