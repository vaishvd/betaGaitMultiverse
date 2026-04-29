from src.config import DATASETS, define_dir


def get_dataset_dirs(dataset):
    cfg = DATASETS[dataset]
    root = cfg["root"]

    dirs = {}
    for key, name in cfg["dirs"].items():
        dirs[key] = define_dir(root, name)

    return dirs


def get_subject_paths(dataset, sub):
    cfg  = DATASETS[dataset]
    dirs = get_dataset_dirs(dataset)

    paths = {}

    # example paths (adapt as needed)
    paths["raw"] = dirs["raw"] / f"sub-{sub}"
    paths["ica"] = dirs["ica"] / f"sub-{sub}_desc-clean_raw.fif"
    paths["gait"] = dirs["gait"] / f"sub-{sub}_gait_cycles.npy"

    paths["events"] = (
        cfg["root"] / "d00_raw" /
        cfg["event_file"].format(sub=sub)
    )

    return paths