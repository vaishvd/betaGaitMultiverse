"""
Active-dataset selector + dataset-agnostic canonical pipeline settings.

Set ACTIVE_DATASET below, or the BETAGAIT_DATASET environment variable,
to "stepup" or "jacobsen" to choose which per-dataset config
(src/config_stepup.py or src/config_jacobsen.py) the whole pipeline
uses. Every attribute of the selected module is re-exported from here,
so existing `from src.config import DATASET, SUBJECTS, ...` imports
throughout scripts/ and src/ keep working unchanged -- switching
datasets never requires touching scripts/ or src/pipeline_steps.py,
only this constant (or the env var).
"""

import os
from pathlib import Path

from src import config_stepup, config_jacobsen

ACTIVE_DATASET = os.environ.get("BETAGAIT_DATASET", "stepup")

_PER_DATASET = {
    "stepup":   config_stepup,
    "jacobsen": config_jacobsen,
}

if ACTIVE_DATASET not in _PER_DATASET:
    raise ValueError(
        f"Unknown ACTIVE_DATASET {ACTIVE_DATASET!r} -- "
        f"expected one of {sorted(_PER_DATASET)}"
    )

_active = _PER_DATASET[ACTIVE_DATASET]

# Re-export every public attribute of the active per-dataset config
# (DATASET, SUBJECTS, MULTIVERSE_SUBJECTS, EEG_FORMAT, EVENT_SOURCE,
# and whatever dataset-specific extras it defines) so `from src.config
# import <name>` works uniformly regardless of which dataset is active.
for _name in dir(_active):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_active, _name)
del _name

# --- Canonical pipeline settings (dataset-agnostic) ---
# ASR (Artifact Subspace Reconstruction) is applied after bad-channel
# interpolation and before average reference.
# Set USE_ASR = True to enable in the canonical pipeline.
# Default is False: ASR attenuated the stance/swing beta difference
# in the multiverse analysis (group t: 1.80 → 0.60) on stepUpAms, so the
# canonical pipeline uses the more conservative non-ASR result by
# default for every dataset.
# See: Mullen et al. 2015 IEEE TBME; Gorjan et al. 2022 J Neural Eng
USE_ASR    = False
ASR_CUTOFF = 30.0   # SD threshold; 20-30 recommended for walking EEG

MULTIVERSE_NAME = "beta_gait_multiverse"


def define_dir(root, *names):
    """Create a directory (parents included) and return it as a Path."""
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

# Group-level pipeline outputs (prepana06's group figure, prepana07's
# group stats, qc_summary.py's aggregated tables) are nested per active
# dataset -- two datasets share the same scripts, but each dataset's
# group-level results must not overwrite the other's.
DIR_PLOTS = define_dir(DIR_RESULTS, "pipeline", ACTIVE_DATASET, "plots")
DIR_QC    = define_dir(DIR_RESULTS, "pipeline", ACTIVE_DATASET, "qc")

# Multiverse outputs (branch ICA cache, COMET's internal working
# directory, final outputs/pkl) are nested per active dataset -- same
# rationale as DIR_PLOTS/DIR_QC above: two datasets share the same
# multiverse scripts, but their branch caches and results must never
# collide or silently overwrite one another.
DIR_MULTIVERSE          = define_dir(DIR_RESULTS, "multiverse", ACTIVE_DATASET)
DIR_MULTIVERSE_OUTPUTS  = define_dir(DIR_MULTIVERSE, "outputs")
DIR_MULTIVERSE_BRANCHES = define_dir(DIR_MULTIVERSE, "branches")

# COMET's own working directory (generated universe_N.py scripts + its
# raw multiverse_results.pkl / multiverse_summary.csv). Passed
# explicitly as Multiverse(path=str(DIR_MULTIVERSE_COMET)) in
# mulana01/02/03 so COMET never falls back to its default
# calling-script-relative location (which is not dataset-aware and was
# the original source of this collision -- see comet.multiverse.
# Multiverse.__init__).
DIR_MULTIVERSE_COMET    = define_dir(DIR_MULTIVERSE, "comet")

# Per-dataset directory trees, keyed for src.paths.get_dataset_dirs().
# Built for every known dataset (not just the active one) so
# get_dataset_dirs("stepup") / get_dataset_dirs("jacobsen") both resolve
# regardless of ACTIVE_DATASET -- e.g. useful for one-off cross-dataset
# scripts or tests.
DATASETS = {
    dataset_name: {
        "root": define_dir(DIR_DATASETS, mod.ROOT_DIRNAME),
        "dirs": mod.DIRS,
    }
    for dataset_name, mod in _PER_DATASET.items()
}
