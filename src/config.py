from pathlib import Path

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
DIR_RESULTS  = define_dir(DIR_PROJ, "results")
DIR_PLOTS    = define_dir(DIR_RESULTS, "plots")

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
            "raw":        "d00_raw",
            "gait_events":"d01_gaitevents",

            "prep":    "d02_prep",

            "clean":      "d03_clean",
            "tfr":        "d04_tfr",
            "ersp":       "d05_ersp",
        },

        "event_file": "events/{sub}.tsv"
    },

}
