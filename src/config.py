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

# --- ERSP normalization: standing-baseline only (2026-08-11) ----------
# Previously selectable via BETAGAIT_NORMALIZATION between "gpm" (dB
# relative to each subject's own whole-gait-cycle mean,
# src.ersp.apply_gpm_normalization) and "standing" (dB relative to the
# standing/rest baseline). GPM was exploratory leftover from an earlier
# "compute both to compare" phase -- stepUpAms was always meant to be
# standing-baseline only, and GPM/standing are mathematically identical
# for the double-stance-vs-swing contrast (a per-subject, per-frequency
# CONSTANT subtracted uniformly across the gait cycle cancels exactly in
# any linear contrast between two subsets of it -- confirmed identical
# t-statistics both times this was checked). Removed entirely; standing-
# baseline is now the only normalization for both datasets. Never
# affected preprocessing, ICA, or the multiverse (multiverse_pipeline.py
# and mulana0*.py never referenced GPM/NORMALIZATION at all).

# --- Permanent band split (was formerly an ACTIVE_VARIANT/BETAGAIT_VARIANT
# toggle between "ref40"/"ref60" -- retired; see git history if that
# mechanism is ever needed again). The reference/canonical pipeline
# (prepana01-07) and the multiverse now each have their own fixed,
# independent TFR ceiling -- they are no longer tied together by a
# single switch:
#   PIPELINE_TFR_FMAX     -- canonical pipeline's TFR upper bound (Hz).
#                            Permanently 60: visualizes beta-gamma
#                            decoupling. The canonical raw low-pass
#                            filter (prepana02_raw2ica.py's H_FREQ=60.0)
#                            is unrelated and unaffected by this constant.
#   MULTIVERSE_TFR_FMAX   -- multiverse's TFR upper bound (Hz).
#                            Permanently 40, unchanged from the original
#                            validated multiverse.
#   MULTIVERSE_LOWPASS_HZ -- multiverse's own fixed lowpass_hz decision
#                            (mulana01_create_multiverse.py's analysis_
#                            template). Changed 40.0 -> 60.0 (2026-08-07):
#                            this was the last remaining reference-vs-
#                            multiverse divergence (ICLabel logic and
#                            gait-event anchors already unified/frozen in
#                            the prior two tasks) -- raised to match the
#                            reference pipeline's own fixed raw lowpass
#                            (prepana02_raw2ica.py's H_FREQ=60.0) so the
#                            reference pipeline becomes reproducible as a
#                            real universe (HP=1, ASR=sd20, IC=balanced,
#                            now lowpass=60) rather than merely
#                            "coincidentally close" to one. The original
#                            40.0-validated multiverse is archived intact
#                            at results/multiverse/stepup_40hz_archive/
#                            before this change, never overwritten.
PIPELINE_TFR_FMAX     = 60.0
MULTIVERSE_TFR_FMAX   = 40.0
MULTIVERSE_LOWPASS_HZ = 60.0

# --- Shared plotting constants (display-only -- never read by any ERSP,
# ROI-weight, or statistical computation; changing these cannot change
# t(16)=5.098 or any other stat). Single source of truth so every ERSP
# heatmap and ERSP-derived topography in the repo (reference-pipeline
# ERSP, its 3 beta topographies, and the multiverse zoom-universe
# heatmaps) shares one colormap and, per plot type, one symmetric limit.
#
#   ERSP_CMAP         -- diverging, zero-centered: red=ERS(+)/blue=ERD(-)/
#                        white=0. Must never be a sequential or rainbow
#                        map (turbo etc.) -- the ERS/ERD sign and the
#                        zero-crossing are the entire point of these plots.
#   ERSP_HEATMAP_VLIM -- symmetric dB limit used ONLY by
#                        mulana04_zoom_universes.py now, to guarantee its
#                        3 zoom-universe panels always share one scale,
#                        trivially, by construction (a fixed constant
#                        rather than a per-run data-driven max). NOT used
#                        by prepana06_plotbetagait.py any more -- that
#                        script now computes its own heatmap/topo limits
#                        per dataset (99th percentile of |value| for that
#                        specific run), since one global constant made
#                        Jacobsen's real range (~+-0.8 dB) render nearly blank
#                        against a limit tuned for stepUpAms (~+-3 dB).
#   BETA_TOPO_VLIM    -- retained for reference/history only; no longer
#                        imported anywhere (prepana06 now computes its
#                        own per-dataset-per-mode beta topo limit the
#                        same way as ERSP_HEATMAP_VLIM above).
ERSP_CMAP         = "RdBu_r"
ERSP_HEATMAP_VLIM = 3.0
BETA_TOPO_VLIM    = 4.0

# --- Centralized static analysis parameters (2026-08-07) -------------------
# Audited every hardcoded analysis parameter across scripts/ and src/ for
# duplicate definitions -- the same class of bug that caused the ICLabel
# rule mismatch between the reference pipeline and the multiverse (the
# reference used to hand-roll its own compound label==brain-and-prob
# condition instead of calling select_ics_by_rule("balanced"), silently
# drifting from the multiverse's logic; see git history same day this
# block was added). Collapsing everything below to one place means there
# is now exactly one location to change any of these. NO VALUE CHANGED
# by this consolidation -- every constant keeps the value it already had
# at its old, now-removed definition site(s).
#
# NOT unified here (each is an intentional, currently-under-review split,
# not a latent bug -- reported separately, left alone on purpose):
#   - PIPELINE_TFR_FMAX (60, reference) vs MULTIVERSE_TFR_FMAX (40,
#     multiverse) -- already separate config constants above.
#   - The reference's fixed raw lowpass filter (60 Hz --
#     scripts/prepana02_raw2ica.py's H_FREQ) vs MULTIVERSE_LOWPASS_HZ,
#     above -- RESOLVED 2026-08-07, both now 60 Hz (see the comment on
#     MULTIVERSE_LOWPASS_HZ above for why and where the 40 Hz multiverse
#     is archived). H_FREQ itself stays a local literal in
#     prepana02_raw2ica.py, not centralized here, since it and
#     MULTIVERSE_LOWPASS_HZ are conceptually two independent fixed
#     decisions that now happen to agree, not one shared constant.
#   - ASR_CUTOFF_BY_MODE["sd3"] (multiverse-only, no reference
#     equivalent) stays in src/multiverse_pipeline.py; only its "sd20"
#     entry now imports ASR_CUTOFF from here instead of re-hardcoding 20.0.
#   - EPOCH_DUR (AutoReject/ICA-fit epoching) vs BASELINE_EPOCH_DUR_S
#     (standing-baseline power epoching, src.ersp.compute_standing_
#     baseline's own default) -- both happened to be 2.0s, but they are
#     different decisions applied to different signals; kept as two
#     separate constants rather than silently merged, exactly to avoid
#     inventing a coupling that was never actually there.
#   - ASR_EDGE_TRIM_S (trims the END of the ASR calibration window) vs
#     BASELINE_EDGE_TRIM_S (trims the END of the standing-baseline
#     window) -- same reasoning: coincidentally both 2.0s, different
#     purposes, kept separate.
#   - Multiverse decision-node VALUES themselves (highpass_hz forks
#     0.5/1.0/2.0, asr_mode forks off/sd3/sd20, iclabel_rule forks
#     conservative/balanced/liberal) stay in mulana01_create_multiverse.py
#     -- that is the multiverse's design-space definition, not a fixed
#     pipeline setting subject to accidental drift.

TARGET_SFREQ             = 250        # Hz -- canonical resample rate (src.pipeline_steps.preprocess_raw)
LINE_FREQ                = 50.0       # Hz -- notch filter (EU mains)
BAD_CHANNEL_ZSCORE       = 3.0        # peak-to-peak z-score threshold for bad-channel interpolation

RANDOM_STATE             = 42         # shared by AutoReject and ICA fit, both pipelines
N_COMPONENTS             = 0.99       # ICA: fraction of variance explained
ICA_METHOD               = "infomax"
ICA_FIT_PARAMS           = {"extended": True}
ICA_DECIM                = 2          # mne.preprocessing.ICA.fit(..., decim=...)
EPOCH_DUR                = 2.0        # s -- fixed-length epochs for AutoReject + ICA fit
AUTOREJECT_N_INTERPOLATE = [1, 2, 4]

ASR_WIN_LEN              = 0.5
ASR_WIN_OVERLAP          = 0.66
ASR_METHOD               = "euclid"
ASR_MEM_SPLITS           = 30         # asrpy transform() memory-chunking -- no effect on output, see src/nodes/asr_node.py
ASR_CALIBRATION_FLOOR_S  = 115.0      # min clean-calibration duration after edge trim (was 120; lowered for Jacobsen's deterministic 118s -- see ASR=20 task)
ASR_EDGE_TRIM_S          = 2.0        # trimmed off the END of the ASR calibration window

ICLABEL_RULE             = "balanced" # reference pipeline's fixed rule (src.ica_utils.select_ics_by_rule)

AMP_THRESH               = 350e-6     # V -- shared gait-cycle / baseline-epoch rejection amplitude
BASELINE_EPOCH_DUR_S     = 2.0        # s -- fixed-length epochs for standing-baseline Morlet power (src.ersp.compute_standing_baseline)
BASELINE_EDGE_TRIM_S     = 2.0        # trimmed off the END of the standing-baseline window (boundary artefact)

BETA_FMIN                = 13.0       # Hz -- beta band, both pipelines (src.ersp.beta_roi_scalar)
BETA_FMAX                = 30.0
TFR_FMIN                 = 8.0        # Hz -- TFR floor, both pipelines (ceiling is PIPELINE_TFR_FMAX / MULTIVERSE_TFR_FMAX above)
TFR_N_CYCLES_DIVISOR     = 2.0        # n_cycles = freq / this, both pipelines
TFR_N_POINTS             = 101        # gait-cycle grid resolution (0-100% in 1% steps)
TFR_EDGE_CROP            = 0.05       # fraction trimmed at each edge post-TFR

ROI_CENTER_CH            = "Cz"       # linear ROI weighting center electrode

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
#
# HISTORY -- ASR was OFF (USE_ASR=False) from the commit that introduced
# it through the reference pipeline's originally validated results
# (stepUpAms t(16)=5.098, Jacobsen t(17)=2.500): ASR attenuated the
# stance/swing beta difference in the multiverse analysis (group t:
# 1.80 -> 0.60) on stepUpAms, so the canonical pipeline used the more
# conservative non-ASR result by default. Those no-ASR results are
# preserved under the "_noasr" filename suffix (e.g.
# stepup_betaphase_stats_noasr.txt) -- NOT "_asr30", since ASR was never
# actually applied at cutoff=30 or any other value for either dataset;
# ASR_CUTOFF=30.0 sat unused below this flag the whole time.
#
# CURRENT -- USE_ASR=True, ASR_CUTOFF=20.0 by default for BOTH datasets:
# the reference pipeline deliberately matches one specific vertex of the
# stepUpAms 27-universe multiverse grid (asr_mode="sd20"), so the
# reference pipeline's own preprocessing reproduces a real multiverse
# universe's numbers as an internal consistency check. Applies
# identically to stepUpAms and Jacobsen (this flag is dataset-agnostic).
# Does not affect the multiverse itself, which sets its own three
# asr_mode arms independently (mulana01_create_multiverse.py) and is
# untouched by this constant.
# See: Mullen et al. 2015 IEEE TBME; Gorjan et al. 2022 J Neural Eng
#
# BETAGAIT_USE_ASR env var override (2026-08-09) -- same env-var-switch
# pattern the old BETAGAIT_NORMALIZATION used before GPM was removed
# (2026-08-11): lets one dataset's reference re-run flip ASR off
# per-invocation (e.g. for stepUpAms's ASR-off/standing-baseline
# reference variant) without changing the shared default, so Jacobsen
# (or any run without the env var set) is completely unaffected.
USE_ASR    = os.environ.get("BETAGAIT_USE_ASR", "true").lower() not in ("false", "0", "no")
ASR_CUTOFF = 20.0   # SD threshold; 20-30 recommended for walking EEG

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
