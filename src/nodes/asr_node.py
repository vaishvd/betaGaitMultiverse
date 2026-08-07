"""ASR (Artifact Subspace Reconstruction) decision node."""

import mne
from asrpy import ASR


def apply_asr_node(
    raw: mne.io.BaseRaw,
    apply: bool,
    calib_raw: mne.io.BaseRaw = None,
    cutoff: float = 30.0,
    win_len: float = 0.5,
    win_overlap: float = 0.66,
    method: str = "euclid",
    mem_splits: int = 30,
) -> mne.io.BaseRaw:
    """
    Optionally apply ASR to a preprocessed raw recording.

    ASR (Mullen et al. 2015, IEEE Trans Biomed Eng) identifies
    high-variance artifact subspaces relative to a clean calibration
    covariance and reconstructs them from neighbouring channels.
    Applied between bad-channel interpolation and average re-referencing,
    on high-pass filtered data.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed, high-pass filtered, bad-channel-interpolated raw
        (pre-reference). Must be preloaded. This is the recording to clean.
    apply : bool
        If True, fit and apply ASR. If False, return raw unchanged
        (the skip branch of the decision node).
    calib_raw : mne.io.BaseRaw or None
        If provided, ASR is calibrated on this segment (e.g. a clean
        standing/resting recording) and then applied to raw. Fitting on a
        separate clean segment rather than the data to be cleaned avoids
        suppressing task-related neural signal in the target recording
        (Mullen et al. 2015). If None, ASR is fitted on raw itself and
        asrpy auto-selects the cleanest windows internally.
    cutoff : float
        SD threshold for artifact rejection. Gorjan et al. 2022
        (J Neural Eng) recommend 20-30 to preserve movement-related
        neural signal in walking EEG; default 30.
    win_len : float
        Calibration window length in seconds.
    win_overlap : float
        Calibration window overlap fraction.
    method : str
        Covariance estimator: 'euclid' or 'riemann'.
    mem_splits : int
        Passed through to asrpy's ASR.transform(): splits the transform
        into this many sequential chunks (state carried across chunks via
        asrpy's internal Zi/cov/carry, so the output is identical
        regardless of the split count -- this is a pure memory/speed
        tradeoff, not an algorithm change). Default raised from asrpy's
        own default of 3 to 30 (2026-08-05): stepUpAms's ~8min recordings
        never needed this, but Jacobsen's ~50min recordings made
        asr.transform's internal per-chunk covariance array (shape
        C x C x chunk_len) exceed available RAM even at mem_splits=3
        (~16-18 GiB peak on this dataset) -- see the ASR=20 reference
        task's investigation. mem_splits=30 divides that peak by ~10x.

    Returns
    -------
    mne.io.BaseRaw
        Cleaned raw (copy) if apply=True, else the input raw unchanged.

    Notes
    -----
    asrpy 0.0.8: ASR.transform does raw.copy() internally and returns the
    cleaned Raw object. The input raw is never mutated.
    """
    if not apply:
        return raw

    asr = ASR(
        sfreq       = raw.info["sfreq"],
        cutoff      = cutoff,
        win_len     = win_len,
        win_overlap = win_overlap,
        method      = method,
    )
    fit_target = calib_raw if calib_raw is not None else raw
    asr.fit(fit_target)
    return asr.transform(raw, mem_splits=mem_splits)   # returns cleaned Raw copy; input unchanged
