from pathlib import Path

# Active dataset and subject list.
# Change DATASET to switch between "stepup" and "splitbelt".
# SUBJECTS will be expanded to the full cohort before batch processing.
DATASET  = "stepup"   # active dataset key
SUBJECTS = [
    "S1", "S2", "S3", "S4", "S7", "S9", "S10", "S11",
    "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S20", "S21", "S23",
]  # full cohort

# Subjects for multiverse analysis.
# S2 excluded: insufficient clean epochs under low high-pass settings
# (0.1 Hz), confirmed by Universe 1 run (15/242 epochs after AutoReject).
MULTIVERSE_SUBJECTS = ["S1", "S3", "S4"]

# --- Canonical pipeline ASR settings ---
# ASR (Artifact Subspace Reconstruction) is applied after bad-channel
# interpolation and before average reference.
# Set USE_ASR = True to enable in the canonical pipeline.
# Default is False: ASR attenuated the stance/swing beta difference
# in the multiverse analysis (group t: 1.80 → 0.60), so the canonical
# pipeline uses the more conservative non-ASR result.
# See: Mullen et al. 2015 IEEE TBME; Gorjan et al. 2022 J Neural Eng
USE_ASR    = False
ASR_CUTOFF = 30.0   # SD threshold; 20-30 recommended for walking EEG

MULTIVERSE_NAME = "beta_gait_multiverse"


def define_dir(root, *names):
    """Creates a directory and ensures it exists."""
    path = root
    for name in names:
        path = path / name
    path.mkdir(parents=True, exist_ok=True)
    return path

# Get the root directory of the repository (parent of 'src')
DIR_PROJ = Path(__file__).resolve().parents[1]

# Define paths for data directories
DIR_DATASETS = define_dir(DIR_PROJ, "datasets")
DIR_SCRIPTS  = define_dir(DIR_PROJ, "scripts")
DIR_RESULTS  = define_dir(DIR_PROJ, "results")
DIR_PLOTS = define_dir(DIR_RESULTS, "pipeline", "plots")
DIR_QC    = define_dir(DIR_RESULTS, "pipeline", "qc")
DIR_MULTIVERSE          = define_dir(DIR_RESULTS, "multiverse")
DIR_MULTIVERSE_OUTPUTS  = define_dir(DIR_RESULTS, "multiverse", "outputs")
DIR_MULTIVERSE_BRANCHES = define_dir(DIR_RESULTS, "multiverse", "branches")

# Dataset-specific directories will be defined in the DATASETS dict below, which allows for flexible handling of multiple datasets with different structures
DATASETS = {

    "splitbelt": {
        "root": define_dir(DIR_DATASETS, "splitBeltFerris"),

        # pipeline structure
        "dirs": {
            "raw":        "d00_raw",
            "montage":    "d00_montage",
            "seg":        "d01_segmented",
            "sigclean":   "d02_sigclean",
            "preica":     "d03_preica",
            "ica":        "d04_ica",
            "gait_cycles":       "d05_gaitcycles",
            "tfr":        "d06_tfr",
            "ersp":       "d07_ersp",
        },

        # dataset-specific parameters
        "event_file": "eeg/sub-{sub}_task-task_events.tsv",
        "condition_start": "B3",
        "condition_end": "End B3",
    },


    "stepup": {
        "root": define_dir(DIR_DATASETS, "stepupAms"),

        "dirs": {
            "qc":         "d00_qc",
            "raw":        "d00_raw",
            "gait_events":"d01_gaitevents",

            "prep":    "d02_prep",

            "clean":      "d03_clean",
            "gaitepochs":        "d04_gaitepochs",
            "ersp":       "d05_ersp",
        },

        "event_file": "events/{sub}.tsv"
    },

}
