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

# --- Baseline: VALIDATED, paper-faithful (2026-08-10) ---
# Baseline: the dataset's own start_standing -> end_standing window (a
# dedicated quiet-standing block the original protocol places between
# start_restEEG and the walking task), used for BOTH (a) ASR
# calibration and (b) ERSP/standing-baseline normalization -- see
# src.pipeline_steps._load_and_annotate_eeglab.
#
# SUPERSEDES the previous start_restEEG + fixed-120s-window baseline
# (restEEG's end marker fires too early to trust, so a fixed duration
# after its start was used as a workaround). start_standing/end_standing
# are the dataset's own explicit, correctly-labeled quiet-standing
# bounds and are the better-matched analogue of the paper's own
# baseline. Re-verified per-subject (2026-08-10, all 18 analysis
# subjects): both markers present exactly once, every window is exactly
# 240.0s, and every window falls strictly between start_restEEG and
# start_easy (never overlapping the walking task) -- see
# results/pipeline/jacobsen/_archive_asr20_restEEG_reference/ for the
# prior baseline's outputs, kept for comparison.
BASELINE_START_VALUE = "start_standing"
BASELINE_END_VALUE   = "end_standing"

# --- Reference-pipeline-only paper-faithful overrides (2026-08-10) -----
# These apply ONLY to the canonical/reference pipeline (prepana02-07),
# not to the multiverse (mulana01/02/03, src.multiverse_pipeline), which
# keeps its own dataset-agnostic 3x3x3 grid (highpass_hz/asr_mode/
# iclabel_rule) identical between stepUpAms and Jacobsen -- see NOTES.md
# for the rationale (two different reference pipelines, one identical
# multiverse, is the intentional design).
#
#   REFERENCE_HIGHPASS_HZ         -- main analysis high-pass (Jacobsen et
#                                     al.'s own 0.2 Hz, OFF the multiverse's
#                                     highpass grid {0.5, 1, 2} -- the
#                                     reference is not meant to be a grid
#                                     vertex here, unlike stepUpAms's
#                                     reference == universe_17).
#   REFERENCE_ICA_FIT_HIGHPASS_HZ -- separate, more aggressive high-pass
#                                     (paper's ICA_bandpass_fmin) applied
#                                     ONLY to the copy of the data ICA is
#                                     fit on (AutoReject epoching + ICA
#                                     decomposition); the resulting ICA
#                                     weights are then applied back to the
#                                     REFERENCE_HIGHPASS_HZ-filtered data,
#                                     not to this more-aggressively-
#                                     filtered copy. Standard practice
#                                     (slow drift removed for a more
#                                     stable decomposition; final data
#                                     keeps the gentler analysis filter).
#   REFERENCE_WARP_ANCHORS_PCT    -- paper's FIXED warp latencies
#                                     (gait_event_newLat=[1,18,50,68,100],
#                                     expressed here as the (A_lto, A_lhs,
#                                     A_rto) percent-of-cycle anchors
#                                     warp_cycle_to_grid()/
#                                     phase_split_indices() already take --
#                                     see src.ersp.load_reference_anchors).
#                                     Imposed, not data-derived: used ONLY
#                                     by the reference pipeline
#                                     (prepana05/06/07); the Jacobsen
#                                     multiverse keeps using the computed
#                                     group-median anchors from
#                                     group_gait_event_anchors_frozen.json,
#                                     same as stepUpAms.
REFERENCE_HIGHPASS_HZ         = 0.2
REFERENCE_ICA_FIT_HIGHPASS_HZ = 2.0
REFERENCE_WARP_ANCHORS_PCT    = (18.0, 50.0, 68.0)   # (A_lto, A_lhs, A_rto)

# --- Documented deviations from Jacobsen et al.'s own pipeline ---------
# The reference pipeline reproduces the paper's preprocessing choices
# with THREE known deviations (all deliberate project decisions, not
# oversights -- record for the manuscript):
#
#   1. ICA: Extended Infomax (src.config.ICA_METHOD="infomax",
#      fit_params={"extended": True}), NOT the paper's AMICA. AMICA has
#      no maintained Python implementation available to this pipeline.
#   2. Line noise: a 50 Hz notch filter (src.config.LINE_FREQ=50.0),
#      NOT the paper's Zapline-plus. Zapline-plus is a MATLAB/EEGLAB
#      plugin with no equivalent wired into this Python pipeline.
#   3. ASR: standard SD-cutoff burst-correction (asrpy, cutoff=20,
#      src.config.ASR_CUTOFF/USE_ASR -- Mullen et al. 2015 style), NOT
#      the paper's channel-rejection-only ASR call (FlatlineCriterion=5,
#      ChannelCriterion=0.8, LineNoiseCriterion=4, burst correction OFF).
#      That specific EEGLAB clean_rawdata channel-rejection algorithm has
#      no Python port in this environment (no MATLAB/EEGLAB available);
#      rather than hand-approximate it, this was an explicit decision
#      (2026-08-10) to keep the existing, already-validated ASR=20
#      burst-correction path unchanged for Jacobsen's reference pipeline.

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
