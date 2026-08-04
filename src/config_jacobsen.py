"""
Per-dataset settings: Jacobsen et al. 2020 outdoor free-walking study
(OpenNeuro ds003039, "A walk in the park?", task-neurCorrYoung).

Encodes everything the shared canonical pipeline needs to treat this
dataset like stepUpAms -- EEGLAB file format, gait events read from the
dataset's own BIDS events.tsv (not detected), and a provisional
fixed-window resting baseline. See the ds003039 inventory report for
full background on every choice documented below.

Pure data module: no filesystem access, no imports from src.config,
so it can be imported by src.config without a circular import.
"""

DATASET      = "jacobsen"
ACCESSION    = "ds003039"
ROOT_DIRNAME = "jacobsen"   # datasets/jacobsen/

# sub-001 .. sub-019 (BIDS zero-padded 3-digit IDs; N=19, all young adults 19-32y)
# All 19 subjects' raw data are downloaded (see
# scripts/setup/00_download_openneuro_dataset.py), but sub-019 is EXCLUDED
# from SUBJECTS below.
#
# KNOWN ISSUE -- sub-019's *_eeg.fdt is truncated AT THE SOURCE on
# OpenNeuro's own S3 mirror (confirmed via HEAD request: the S3
# object's Content-Length exactly matches our local file, so this is
# not a download error). mne.io.read_raw_eeglab(..., preload=True)
# raises "Incorrect number of samples (2142427 != 24999978)" -- the
# .set header's declared sample count doesn't match the actual .fdt
# payload. Re-downloading does not help, so it is excluded outright
# (usable N = 18) rather than left to fail per-subject at runtime.
SUBJECTS = [f"{i:03d}" for i in range(1, 19)]   # sub-001..sub-018 (sub-019 excluded, see above)

# NOTE -- sub-018's raw .set has 6 extra channels (Left/Right_
# Accelerometer_X/Y/Z) not listed in its own *_channels.tsv (a BIDS
# metadata omission specific to this subject; every other subject's
# .set matches its channels.tsv exactly, 64 EEG + 3 MISC = 67). These
# aren't dropped by _load_and_annotate_eeglab's channels.tsv-driven
# MISC filter (src.pipeline_steps), but they get no electrode position
# (absent from electrodes.tsv too) and so are removed automatically by
# the existing no-position-channel cleanup in fit_ica()/apply_ica() --
# verified, no code change needed.

# Multiverse cohort not yet validated for this dataset -- the canonical
# pipeline (prepana01-07) has only been run end-to-end on a 3-subject
# test slice so far (see Step 5 verification). Provisionally the full
# cohort; revisit once ICA/AutoReject QC has been reviewed per subject,
# the way S2/S10/S21 were excluded for stepUpAms.
MULTIVERSE_SUBJECTS = SUBJECTS

# --- Recording / file format ---
EEG_FORMAT = "eeglab"          # .set/.fdt, loaded via mne.io.read_raw_eeglab
TASK_NAME  = "neurCorrYoung"   # single BIDS task; one continuous ~50 min recording per subject
EEG_SFREQ_NATIVE  = 500.0      # Hz, matches sub-*_eeg.json SamplingFrequency and events.tsv's "sample" column
MONTAGE_SOURCE    = "electrodes_tsv"  # positions read from BIDS electrodes.tsv, NOT a standard template
EEG_REFERENCE_RAW = "common"   # generic online reference per BIDS eeg.json; pipeline re-references to average regardless, so this is inert

# --- Gait event source ---
# Gait events (LeftHS/RightHS/LeftTO/RightTO) are pre-computed by the
# dataset authors from foot-worn accelerometers (~2 ms reported
# accuracy) and distributed as rows in events.tsv -- prepana01 reads
# them directly instead of detecting from motion capture.
#
# LABEL INCONSISTENCY -- sub-018 uses a different naming convention for
# the same four event types ("IC_Left"/"IC_Right"/"TO_Left"/"TO_Right",
# i.e. Initial-Contact/Toe-Off) instead of every other subject's
# "LeftHS"/"RightHS"/"LeftTO"/"RightTO". Discovered because the fixed
# single-string mapping silently produced 0 events for sub-018 (an
# empty, 0-byte cycles.tsv). Each code below now lists every synonym
# seen across subjects; src.gait_cycles.load_gait_events_from_tsv
# matches whichever one is actually present in a given subject's file.
EVENT_SOURCE = "events_tsv"
GAIT_EVENT_VALUES = {
    "LHS": ["LeftHS", "IC_Left"],
    "RHS": ["RightHS", "IC_Right"],
    "LTO": ["LeftTO", "TO_Left"],
    "RTO": ["RightTO", "TO_Right"],
}

# --- Segment: steady-state even-terrain walking WITHOUT button presses ---
# (excludes the *_button variants -- see inventory report step 3c)
SEGMENT_START_VALUE = "start_easy"
SEGMENT_END_VALUE   = "end_easy"

# --- Baseline: VALIDATED ---
# Baseline: 120 s standing window from start_restEEG onset. The
# restEEG start/end markers in ds003039 are mislabeled (end fires
# early), but the standing-baseline DATA is present; supervisor-
# confirmed that the 120 s immediately after start_restEEG is the
# correct standing baseline. Baseline is a divisive normalization
# applied identically to both phases of the paired double-stance-vs-
# swing contrast, so it cancels from the contrast; it affects only
# absolute dB levels.
#
# Per-subject validated (all 18 analysis subjects: a full 120 s of
# continuous recording exists after start_restEEG onset, and that
# window does not run into the walking task) -- see
# scripts/diag_jacobsen_baseline_check.py and
# results/pipeline/jacobsen/qc/baseline_120s_check.txt.
BASELINE_START_VALUE = "start_restEEG"
BASELINE_DURATION_S  = 120.0

# --- Annotation names the shared pipeline crops on (see src.pipeline_steps) ---
# Both reuse the exact names stepUpAms's pipeline_steps/prepana04-07/
# multiverse_pipeline already crop by, so no downstream script branches
# on dataset.
BASELINE_ANNOTATION = "STAND"   # <- BASELINE_START_VALUE + BASELINE_DURATION_S window
WALK_ANNOTATION     = "CS"      # <- SEGMENT_START_VALUE .. SEGMENT_END_VALUE window

# --- Known limitation: turns ---
# ds003039 provides no turn markers, heading, or route/GPS data -- only
# 3 MISC accelerometer channels (x_dir/y_dir/z_dir), used by the
# original authors solely to classify gait-*initiation* direction, not
# for continuous heading. Turns within the "easy" outdoor walking bout
# cannot be detected or excluded from this dataset. Accepted limitation.
TURNS_DETECTABLE = False

# Directory layout, relative to datasets/jacobsen/ -- identical keys to
# config_stepup.DIRS so src.paths.get_dataset_dirs() and every shared
# script work unchanged regardless of which dataset is active.
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
