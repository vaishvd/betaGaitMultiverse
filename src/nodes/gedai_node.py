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
    broadband_noise_multiplier: float = 6.0,
    spectral_noise_multiplier: float = 3.0,
    wavelet_type: str = "haar",
    wavelet_level: int = 5,
    wavelet_low_cutoff: float = 2.0,
    sensai_method: str = "optimize",
) -> mne.io.BaseRaw:
    """
    Optionally apply two-stage GEDAI to a preprocessed raw recording.

    Implements the recommended two-stage GEDAI pipeline per the official
    GEDAI spectral tutorial (neurotuning.github.io/gedai):

      Stage 1 — broadband, conservative (noise_multiplier=6.0):
        Fits a single broadband Gedai instance to remove only large
        artifacts. High noise_multiplier preserves neural signal.

      Stage 2 — spectral, band-specific (Haar wavelet, level 5):
        Fits a second Gedai instance on the stage-1 output using
        frequency-band decomposition, allowing fine-grained removal
        of band-limited artifacts while leaving oscillatory signal
        intact. Epoch duration/overlap are omitted (auto-adjusted
        internally by the spectral fit).

    Applied after bad-channel interpolation and before average
    re-referencing. Montage is temporarily projected to standard 10-20
    for leadfield covariance compatibility, then original channel info
    is restored.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed, filtered, bad-channel-interpolated raw
        (pre-reference). Must be preloaded.
    apply : bool
        If True, run the two-stage pipeline. If False, return raw
        unchanged (the skip branch of the decision node).
    duration : float
        Epoch duration in seconds for broadband stage covariance.
    overlap : float
        Epoch overlap fraction (0-1) for broadband stage.
    broadband_noise_multiplier : float
        Rejection threshold for stage 1 (conservative). Default 6.0.
    spectral_noise_multiplier : float
        Rejection threshold for stage 2 (band-specific). Default 3.0.
    wavelet_type : str
        Wavelet family for stage-2 decomposition. Default "haar".
    wavelet_level : int
        Decomposition depth for stage-2. Default 5.
    wavelet_low_cutoff : float
        Low-frequency cutoff (Hz) for stage-2 wavelet bands. Default 2.0.
    sensai_method : str
        Regularisation selection method for both stages. Default "optimize".

    Returns
    -------
    mne.io.BaseRaw
        Two-stage GEDAI-cleaned raw if apply=True, else input unchanged.
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

    # ── Stage 1: broadband, conservative ───────────────────────
    gedai_bb = Gedai()
    gedai_bb.fit_raw(
        raw_work,
        duration=duration,
        overlap=overlap,
        reject_by_annotation=False,
        reference_cov="leadfield",
        sensai_method=sensai_method,
        noise_multiplier=broadband_noise_multiplier,
        verbose=False,
    )
    raw_bb = gedai_bb.transform_raw(
        raw_work, duration=duration, overlap=overlap, verbose=False
    )

    # ── Stage 2: spectral, band-specific ───────────────────────
    gedai_sp = Gedai(
        wavelet_type=wavelet_type,
        wavelet_level=wavelet_level,
        wavelet_low_cutoff=wavelet_low_cutoff,
    )
    gedai_sp.fit_raw(
        raw_bb,
        reject_by_annotation=False,
        reference_cov="leadfield",
        sensai_method=sensai_method,
        noise_multiplier=spectral_noise_multiplier,
        verbose=False,
    )
    raw_work_clean = gedai_sp.transform_raw(raw_bb, verbose=False)

    # ── Restore original channel info ───────────────────────────
    # Copy cleaned data back into the original raw object so
    # channel names, positions, and annotations are preserved.
    raw_clean = raw.copy()
    raw_clean._data[:] = raw_work_clean.get_data()

    try:
        n_bb = len(gedai_bb.wavelets_fits)
        bb_str = ", ".join(
            "%d-%d Hz (thr=%.3f)" % (wf["fmin"], wf["fmax"], wf["threshold"])
            for wf in gedai_bb.wavelets_fits
        )
        n_sp = len(gedai_sp.wavelets_fits)
        sp_str = ", ".join(
            "%d-%d Hz (thr=%.3f)" % (wf["fmin"], wf["fmax"], wf["threshold"])
            for wf in gedai_sp.wavelets_fits
        )
        print("  GEDAI stage-1 broadband: %d band(s) -- %s" % (n_bb, bb_str))
        print("  GEDAI stage-2 spectral : %d band(s) -- %s" % (n_sp, sp_str))
    except AttributeError:
        print("  GEDAI two-stage applied (band details unavailable)")

    return raw_clean
