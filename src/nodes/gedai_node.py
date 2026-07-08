"""
GEDAI (Generalized Eigenvalue De-Artifacting Instrument) decision node.

GEDAI is an unsupervised artifact removal method based on leadfield
covariance modelling. It separates brain signals from noise using
generalized eigenvalue decomposition.

Reference: neurotuning.github.io/gedai
"""

import mne


def apply_gedai_node(
    raw: mne.io.BaseRaw,
    apply: bool = True,
    duration: float = 2.0,
    overlap: float = 0.5,
    noise_multiplier: float = 3.0,
    sensai_method: str = "gridsearch",
) -> mne.io.BaseRaw:
    """
    Optionally apply GEDAI to a preprocessed raw recording.

    GEDAI (Hecker et al. 2024, neurotuning.github.io/gedai) uses
    generalized eigenvalue decomposition with a leadfield reference
    covariance to suppress non-brain artifacts while preserving
    neural signal structure. Applied after bad-channel interpolation
    and before average re-referencing.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed, filtered, bad-channel-interpolated raw
        (pre-reference). Must be preloaded.
    apply : bool
        If True, fit and apply GEDAI. If False, return raw unchanged
        (the skip branch of the decision node).
    duration : float
        Epoch duration in seconds used for covariance estimation.
    overlap : float
        Epoch overlap fraction (0–1).
    noise_multiplier : float
        Threshold multiplier for component rejection (higher = less
        aggressive). Default 3.0.
    sensai_method : str
        Method for selecting the regularisation parameter.
        "gridsearch" (default) or "fixed".

    Returns
    -------
    mne.io.BaseRaw
        GEDAI-cleaned raw if apply=True, else the input raw unchanged.
    """
    if not apply:
        print("  GEDAI skipped (apply=False)")
        return raw

    from gedai import Gedai

    # ── Project to standard 10-20 for GEDAI compatibility ──────
    # GEDAI's leadfield covariance is built on standard 10-20.
    # Channels outside that set are marked bad and interpolated
    # so GEDAI receives a conformant montage. Original channel
    # info is restored after the transform.

    montage_1020 = mne.channels.make_standard_montage("standard_1020")
    ch_names_1020 = set(montage_1020.ch_names)

    # Work on a copy so the caller's raw is never mutated
    raw_work = raw.copy()
    raw_work.set_montage("standard_1020", on_missing="ignore")

    non_1020 = [ch for ch in raw_work.ch_names
                if ch not in ch_names_1020]
    if non_1020:
        raw_work.info["bads"] = non_1020
        raw_work.interpolate_bads(reset_bads=True)
        print(f"  GEDAI: interpolated {len(non_1020)} non-10-20 "
              f"channel(s) for montage compatibility")

    # ── Fit and transform ───────────────────────────────────────
    gedai = Gedai()
    gedai.fit_raw(
        raw_work,
        duration=duration,
        overlap=overlap,
        reject_by_annotation=False,
        reference_cov="leadfield",
        sensai_method=sensai_method,
        noise_multiplier=noise_multiplier,
        verbose=False,
    )
    raw_work_clean = gedai.transform_raw(
        raw_work,
        duration=duration,
        overlap=overlap,
        verbose=False,
    )

    # ── Restore original channel info ───────────────────────────
    # Copy cleaned data back into the original raw object so
    # channel names, positions, and annotations are preserved.
    raw_clean = raw.copy()
    raw_clean._data[:] = raw_work_clean.get_data()

    n_bands = len(gedai.wavelets_fits)
    thresholds = ", ".join(
        f"{wf['fmin']:.0f}-{wf['fmax']:.0f} Hz "
        f"(thr={wf['threshold']:.3f})"
        for wf in gedai.wavelets_fits
    )
    print(f"  GEDAI applied: {n_bands} band(s) — {thresholds}")
    return raw_clean
