"""
Per-dataset settings: stepUpAms (VU Amsterdam level-walking / step-up
study, BrainVision EEG + optical motion capture).

Relocated unchanged from the former monolithic src/config.py -- no
behaviour change. This module is pure data (no filesystem access, no
imports from src.config) so it can be imported by src.config without
creating a circular import.
"""

DATASET      = "stepup"
ROOT_DIRNAME = "stepup"   # datasets/stepup/

# Full cohort.
SUBJECTS = [
    "S1", "S2", "S3", "S4", "S7", "S9", "S10", "S11",
    "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S20", "S21", "S23",
]

# Subjects for multiverse analysis.
# Excluded:
#   S10 — ICA failure: 1/38 brain ICs (brain_frac=0.026), signal quality insufficient
#   S21 — AutoReject failure: only 12/243 clean epochs, extreme artifact contamination (bad P1)
#   S2 — AutoReject failure with only 12/243 clean epochs.
MULTIVERSE_SUBJECTS = [
    "S1", "S3", "S4", "S7", "S9", "S11",
    "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S20", "S23",
]

# --- Recording / file format ---
EEG_FORMAT = "brainvision"        # .vhdr/.eeg/.vmrk, loaded via mne.io.read_raw_brainvision
TASK_STAND = "STAND"              # BIDS task suffix for the quiet-standing recording
TASK_WALK  = "CS"                 # BIDS task suffix for the walking recording
MONTAGE_SOURCE    = "standard_1005"  # assigned programmatically; no electrodes.tsv in this dataset
EEG_REFERENCE_RAW = "n/a"            # not specified in this dataset's BIDS eeg.json; pipeline re-references to average regardless

# --- Gait event source ---
EVENT_SOURCE = "mocap"   # prepana01 detects HS/TO from motion-capture heel/pelvis markers

# --- Annotation names the shared pipeline crops on (see src.pipeline_steps) ---
BASELINE_ANNOTATION = "STAND"   # quiet-standing segment, used directly as the ERSP baseline
WALK_ANNOTATION     = "CS"      # walking segment gait cycles are extracted from

# Directory layout, relative to datasets/stepup/
DIRS = {
    "qc":          "d00_qc",
    "raw":         "d00_raw",
    "gait_events": "d01_gaitevents",
    "prep":        "d02_prep",
    "clean":       "d03_clean",
    "gaitepochs":  "d04_gaitepochs",
    "ersp":        "d05_ersp",
    "roi_topo":    "d05_roi_topo",
}
