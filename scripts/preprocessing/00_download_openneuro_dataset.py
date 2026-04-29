from src.config import DATASETS
from src.paths import get_dataset_dirs
import openneuro as on

DATASET = "splitbelt"

cfg  = DATASETS[DATASET]
dirs = get_dataset_dirs(DATASET)

on.download(
    dataset="ds004475",
    target_dir=dirs["raw"],
    exclude=["**/anat", "**/headmodel"],
)