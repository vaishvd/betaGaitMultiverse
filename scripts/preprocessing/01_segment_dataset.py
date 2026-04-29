"""01_segment_dataset.py
Segment pre-adaptation phase from gait dataset.
The dataset consists of a 40-minute recording":

    0–15 min → pre-adaptation
    15–30 min → split-belt adaptation
    30–40 min → post-adaptation

We are only interested in the 5-min medium speed recording (B3)during the 15 min "pre-adaptation" phase. The gait events are already in the dataset for each subject "sub-S18_task-task_events.tsv".

"""

from src.paths import get_subject_paths, get_dataset_dirs
from src.config import DATASETS
from src.bids import load_raw_bids
from src.events import load_events
from src.segmentation import find_segment, crop_raw


DATASET = "splitbelt"
SUBJECTS = ["S18"]

BUFFER = 5

cfg  = DATASETS[DATASET]
dirs = get_dataset_dirs(DATASET)

OUTPUT_DIR = dirs["seg"]

for sub in SUBJECTS:

    print(f"\nProcessing subject {sub}")

    paths = get_subject_paths(DATASET, sub)

    raw = load_raw_bids(sub, "task", "eeg", dirs["raw"])

    events_df = load_events(paths["events"])

    start_time, end_time = find_segment(
        events_df,
        cfg["condition_start"],
        cfg["condition_end"],
    )

    raw_seg, _, _ = crop_raw(raw, start_time, end_time, BUFFER)

    out = OUTPUT_DIR / f"sub-{sub}_seg_raw.fif"
    raw_seg.save(out, overwrite=True)

    print(f"Saved → {out}")