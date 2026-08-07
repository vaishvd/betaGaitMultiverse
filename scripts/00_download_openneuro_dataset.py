# Jacobsen-specific: downloads ds003039 only, not reusable for stepUpAms.

from src.config import DATASETS
from src.paths import get_dataset_dirs
import openneuro as on

DATASET = "jacobsen"

cfg  = DATASETS[DATASET]
dirs = get_dataset_dirs(DATASET)

on.download(
    dataset="ds003039",
    target_dir=dirs["raw"],
)