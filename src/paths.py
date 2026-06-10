from pathlib import Path
from src.config import DATASETS, define_dir


def get_dataset_dirs(dataset):
    """Return directory paths for a given dataset as a dict of Path objects."""

    cfg = DATASETS[dataset]
    root = cfg["root"]
    dirs = {}
    for key, folder in cfg["dirs"].items():
        dirs[key] = define_dir(root, folder)
    return dirs
