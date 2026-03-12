"""segment_dataset.py
Segment pre-adaptation phase from gait dataset.
The dataset consists of a 40-minute recording":

    0–15 min → pre-adaptation
    15–30 min → split-belt adaptation
    30–40 min → post-adaptation

We are only interested in the 15 min "pre-adaptation" phase. The gait events are already in the dataset for each subject "sub-S18_task-task_events.tsv".

"""

from pathlib import Path
from src.config import DIR_RAWDATA, DIR_DATA
from src.bids import get_subjects, load_raw_bids
from src.events import load_events
from src.segmentation import find_segment, crop_raw


TASK = "task"
DATATYPE = "eeg"

START_MARKER = "B1"
END_MARKER = "End B3"

BUFFER = 5

OUTPUT_DIR = DIR_DATA / "segmented"
OUTPUT_DIR.mkdir(exist_ok=True)

SUBJECTS = ["S18"]   # run only this subject

subjects = SUBJECTS

print(f"Found {len(subjects)} subjects")


for subject in subjects:

    print(f"\nProcessing subject {subject}")

    raw = load_raw_bids(subject, TASK, DATATYPE, DIR_RAWDATA)

    events_file = (
        DIR_RAWDATA
        / f"sub-{subject}"
        / "eeg"
        / f"sub-{subject}_task-task_events.tsv"
    )

    events_df = load_events(events_file)
    events_df = load_events(events_file)

    print(events_df.columns)
    print(events_df["value"].unique())

    start_time, end_time = find_segment(events_df, START_MARKER, END_MARKER)

    raw_seg, crop_start, crop_end = crop_raw(
        raw,
        start_time,
        end_time,
        BUFFER,
    )

    output_file = OUTPUT_DIR / f"sub-{subject}_preadapt_raw.fif"

    raw_seg.save(output_file, overwrite=True)

    print(f"Saved → {output_file}")