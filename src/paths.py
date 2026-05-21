from pathlib import Path
from src.config import DATASETS, define_dir


def get_dataset_dirs(dataset):

    cfg = DATASETS[dataset]
    root = cfg["root"]
    dirs = {}
    for key, folder in cfg["dirs"].items():
        dirs[key] = define_dir(root, folder)
    return dirs


def get_subject_paths(dataset, sub):
    cfg  = DATASETS[dataset]
    dirs = get_dataset_dirs(dataset)

    root = cfg["root"]

    paths = {}

    # RAW DATA + EVENTS (d00)

    raw_dir = root / cfg["dirs"]["raw"]

    paths["raw_dir"] = raw_dir

    paths["events"] = raw_dir / f"sub-{sub}" / cfg["event_file"].format(sub=sub)


    # ICA CLEANED DATA (d04)

    ica_dir = root / cfg["dirs"]["ica"]

    paths["ica"] = ica_dir / f"sub-{sub}_desc-clean_raw.fif"

    # GAIT CYCLES (d05)

    gait_dir = root / cfg["dirs"]["gait"]

    paths["gait_cycles"]   = gait_dir / f"sub-{sub}_gait_cycles.npy"
    paths["gait_sfreq"]    = gait_dir / f"sub-{sub}_sfreq.npy"
    paths["gait_durations"] = gait_dir / f"sub-{sub}_durations.npy"


    paths["tfr"] = root / cfg["dirs"].get("tfr", "d06_tfr") / f"sub-{sub}_tfr_beta.npy"
    paths["ersp"] = root / cfg["dirs"].get("ersp", "d07_ersp") / f"sub-{sub}_ersp_beta.npy"

    return paths