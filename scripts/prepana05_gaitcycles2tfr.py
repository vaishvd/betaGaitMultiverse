"""
ana05_gaitcycles2tfr.py
=======================

# ERSP = mean over cycles of per-cycle log ratio to standing baseline.
# Formulation: ersp_avg = mean_k( 10*log10( power_k / baseline_standing ) )
# Standing baseline: mean Morlet power during quiet standing (task-STAND),
# averaged over accepted standing epochs and time, per channel per frequency.
# This approach follows Seeber et al. 2015 J Neurosci and
# Wagner et al. 2012 Neuroimage.
# Per-cycle log ratio before averaging follows Makeig et al. 1993 and
# Delorme & Makeig 2004 J Neurosci Methods.

Input
-----
d03_clean/
    sub-{sub}_desc-icaClean_concat_raw.fif   (ICA-cleaned concatenated STAND+CS raw)

d04_gaitepochs/
    sub-{sub}_gait_segments.npy       object array (n_cycles,), each (n_ch, n_samp_i)
    sub-{sub}_gait_sfreq.npy          scalar float

Output
------
d05_ersp/
    sub-{sub}_ersp_beta.npy           shape (n_ch, n_freqs, N_POINTS)
"""

import json
import sys

import numpy as np
import pandas as pd
import mne

from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS, PIPELINE_TFR_FMAX
from src.qc import log_qc
from src.resume import stage_already_done
from src.spatial_filter import linear_roi_weights, apply_linear_roi, plot_weight_topography
from src.ersp import warp_cycle_to_grid, phase_split_indices, compute_standing_baseline

FREQS    = np.arange(8, int(PIPELINE_TFR_FMAX) + 1, dtype=float)  # 8-60 Hz, permanent (alpha-beta-gamma)
N_CYCLES = FREQS / 2.0                      # frequency-dependent wavelet width
N_POINTS = 101  # 0-100% gait cycle in 1% steps; below native resolution (~248 samples at 250 Hz), avoids upsampling artefacts
EDGE_CROP = 0.05                            # fraction trimmed at each edge post-TFR

# Note on edge artefacts: at 13 Hz with n_cycles=6.5, the wavelet window
# is ~500 ms. Mean cycle duration is ~992 ms, so edge contamination extends
# ~25% of the cycle at each end. The 5% edge crop removes only ~50 ms and
# does not fully eliminate this. Interpretation of gait cycle positions
# 0-25% and 75-100% should be treated with caution at frequencies <= 16 Hz.
# This is flagged as a limitation; multiverse will fork on EDGE_CROP.
# See: Cohen 2014 "Analyzing Neural Time Series Data", MIT Press, Ch. 13

AMP_THRESH_BASELINE = 350e-6
AMP_THRESH_CYCLE    = 350e-6

dirs = get_dataset_dirs(DATASET)

CLEAN_DIR     = dirs["clean"]
GAITEPOCH_DIR = dirs["gaitepochs"]
ERSP_DIR      = dirs["ersp"]
QC_DIR        = dirs["qc"]
ROI_TOPO_DIR  = dirs["roi_topo"]


# --- Group-median gait-event anchors (computed once, pooled across every
# subject's kept cycles, before the per-subject loop) ---
# Pooling all cycles from all subjects together (rather than taking the
# median of each subject's median) is the simpler option and is used here;
# it weights every cycle equally instead of every subject equally.
# These anchors define where LTO/LHS/RTO fall on the common 0-100% gait
# cycle grid. They are used below (a) to warp each cycle's power time
# series with a 4-segment piecewise-linear map instead of a single
# endpoint-only stretch, and (b) to define the double-stance/swing phase
# windows on the resulting fixed grid.
_f_lto, _f_lhs, _f_rto = [], [], []
for _subject in SUBJECTS:
    _meta_path = GAITEPOCH_DIR / f"sub-{_subject}_cycles_kept.tsv"
    if not _meta_path.exists():
        continue
    _meta = pd.read_csv(_meta_path, sep="\t")
    if len(_meta) == 0:
        continue
    _dur = _meta["rhs_end_s"] - _meta["rhs_start_s"]
    _f_lto.append(((_meta["lto_s"] - _meta["rhs_start_s"]) / _dur).values)
    _f_lhs.append(((_meta["lhs_s"] - _meta["rhs_start_s"]) / _dur).values)
    _f_rto.append(((_meta["rto_s"] - _meta["rhs_start_s"]) / _dur).values)

if not _f_lto:
    raise RuntimeError(
        "No sub-*_cycles_kept.tsv files found in GAITEPOCH_DIR -- "
        "cannot compute group gait-event anchors. Run prepana04 first."
    )

A_lto = float(np.median(np.concatenate(_f_lto))) * 100
A_lhs = float(np.median(np.concatenate(_f_lhs))) * 100
A_rto = float(np.median(np.concatenate(_f_rto))) * 100

n_subjects_pooled = len(_f_lto)
n_cycles_pooled   = sum(len(a) for a in _f_lto)

assert 0 < A_lto < A_lhs < A_rto < 100, \
    f"Group anchors are not monotonic: A_lto={A_lto}, A_lhs={A_lhs}, A_rto={A_rto}"

print(f"Group-median gait-event anchors "
      f"(pooled across {n_subjects_pooled} subjects, {n_cycles_pooled} cycles):")
print(f"  A_lto = {A_lto:.2f}%   A_lhs = {A_lhs:.2f}%   A_rto = {A_rto:.2f}%")

with open(ERSP_DIR / "group_gait_event_anchors.json", "w") as _f:
    json.dump({
        "A_lto_pct":         A_lto,
        "A_lhs_pct":         A_lhs,
        "A_rto_pct":         A_rto,
        "n_subjects_pooled": n_subjects_pooled,
        "n_cycles_pooled":   n_cycles_pooled,
    }, _f, indent=2)
print(f"  Saved -> group_gait_event_anchors.json")

# Optional: restrict the PER-SUBJECT loop below to a single subject (see
# prepana02_raw2ica.py's identical mechanism) -- but the group-anchor
# pooling above always uses the full SUBJECTS list regardless, since the
# anchors must be pooled across every subject's cycles, not just the one
# being processed in this invocation. Every single-subject invocation
# harmlessly recomputes and rewrites the same (deterministic) anchors
# file before processing its own subject.
LOOP_SUBJECTS = SUBJECTS
if len(sys.argv) > 1:
    LOOP_SUBJECTS = [sys.argv[1]]

for subject in LOOP_SUBJECTS:
    try:
        print(f"ERSP: sub-{subject}")

        clean_fif  = CLEAN_DIR / f"sub-{subject}_desc-icaClean_concat_raw.fif"
        seg_path   = GAITEPOCH_DIR / f"sub-{subject}_gait_segments.npy"

        out_ds      = ERSP_DIR / f"sub-{subject}_ersp_double_stance.npy"
        out_sw      = ERSP_DIR / f"sub-{subject}_ersp_swing.npy"
        out_beta    = ERSP_DIR / f"sub-{subject}_ersp_beta.npy"
        out_weights = ERSP_DIR / f"sub-{subject}_roi_weights.npy"

        if stage_already_done(
            [out_ds, out_sw, out_beta, out_weights],
            inputs=[clean_fif, seg_path],
            validate=lambda: np.load(out_beta),
        ):
            print(f"  Already complete -- skipping sub-{subject}")
            continue

        # Load ICA-cleaned concatenated raw; preload=True so STAND segment can be extracted
        concat_raw   = mne.io.read_raw_fif(clean_fif, preload=True, verbose=False)
        ch_names_ref = list(concat_raw.ch_names)
        print(f"  Reference channels from clean raw: {len(ch_names_ref)}")

        # Load Gait Segments

        sfreq_path = GAITEPOCH_DIR / f"sub-{subject}_gait_sfreq.npy"

        if not seg_path.exists():
            print(f"  Gait segments not found: {seg_path.name} -- skipping.")
            continue

        gait_segments = np.load(seg_path, allow_pickle=True)
        gait_sfreq    = float(np.load(sfreq_path))

        print(f"  Gait segments: {len(gait_segments)}  sfreq={gait_sfreq:.0f} Hz")

        # Per-cycle event metadata (lto_s/lhs_s/rto_s), row-aligned with
        # gait_segments by construction in ana04 (both built in the same
        # loop over kept cycles). Needed below to warp each cycle's power
        # time series onto the group-median event anchors.
        cycles_meta = pd.read_csv(
            GAITEPOCH_DIR / f"sub-{subject}_cycles_kept.tsv", sep="\t"
        )
        if len(cycles_meta) != len(gait_segments):
            raise RuntimeError(
                f"cycles_kept.tsv has {len(cycles_meta)} rows but gait_segments has "
                f"{len(gait_segments)} cycles -- ana04 meta/segment misalignment."
            )

        # Standing baseline -- extract from ICA-cleaned concatenated raw using annotation

        stand_annot = [a for a in concat_raw.annotations if a["description"] == "STAND"]
        if len(stand_annot) != 1:
            raise RuntimeError(f"Expected 1 STAND annotation, found {len(stand_annot)}")

        stand_start = float(stand_annot[0]["onset"])
        stand_stop  = float(stand_annot[0]["onset"]) + float(stand_annot[0]["duration"])
        stand_stop  = min(stand_stop, concat_raw.times[-1])   # clamp to recording end
        raw_stand   = concat_raw.copy().crop(stand_start, stand_stop)

        # Trim last 2 s to remove boundary artifact from crop operation
        stand_tmax = raw_stand.times[-1] - 2.0
        if stand_tmax <= 0:
            raise RuntimeError(
                f"sub-{subject}: standing segment too short after trimming "
                f"({raw_stand.times[-1]:.1f} s)"
            )
        raw_stand = raw_stand.crop(tmax=stand_tmax)

        stand_ch_names = list(raw_stand.ch_names)

        baseline_power = compute_standing_baseline(
            raw_stand, FREQS, N_CYCLES,
            edge_crop=EDGE_CROP, amp_thresh=AMP_THRESH_BASELINE,
        )

        print(f"  Baseline shape: {baseline_power.shape}  "
              f"range=[{baseline_power.min():.4e}, {baseline_power.max():.4e}]")

        # Gait segments and standing share the same source file; channels are identical
        ref_idx = [ch_names_ref.index(ch) for ch in stand_ch_names if ch in ch_names_ref]

        # Per cycle TFR

        tfr_cycles = []
        n_rej     = 0
        n_rej_evt = 0

        for i, seg in enumerate(gait_segments):   # seg: (n_ch_ref, n_samp_i)

            seg_ch = seg[ref_idx]   # select and reorder to match standing channels

            if np.abs(seg_ch).max() > AMP_THRESH_CYCLE:
                n_rej += 1
                continue

            power = mne.time_frequency.tfr_array_morlet(
                seg_ch[np.newaxis],
                sfreq=gait_sfreq,
                freqs=FREQS,
                n_cycles=N_CYCLES,
                output="power",
                zero_mean=True,
                verbose=False,
            )[0]   # (n_ch, n_freqs, n_samp_i)

            crop = int(EDGE_CROP * power.shape[-1])
            if crop > 0:
                power = power[..., crop:-crop]

            # Full-event (4-segment) piecewise-linear time warp (src.ersp,
            # shared with the multiverse pipeline). Locate this cycle's own
            # LTO/LHS/RTO sample positions within the (edge-cropped) power
            # array, crop-adjusted the same way as `power` above.
            row     = cycles_meta.iloc[i]
            i_start_samp = int(round(float(row["rhs_start_s"]) * gait_sfreq))
            lto_idx = int(round(float(row["lto_s"]) * gait_sfreq)) - i_start_samp - crop
            lhs_idx = int(round(float(row["lhs_s"]) * gait_sfreq)) - i_start_samp - crop
            rto_idx = int(round(float(row["rto_s"]) * gait_sfreq)) - i_start_samp - crop

            try:
                resampled = warp_cycle_to_grid(
                    power, lto_idx, lhs_idx, rto_idx,
                    anchors=(A_lto, A_lhs, A_rto), n_points=N_POINTS,
                )   # (n_ch, n_freqs, N_POINTS)
            except ValueError:
                n_rej_evt += 1
                continue

            tfr_cycles.append(resampled)

        print(f"  Gait cycles rejected by amplitude    : {n_rej}")
        print(f"  Gait cycles rejected by event order  : {n_rej_evt}")
        print(f"  Gait cycles accepted                 : {len(tfr_cycles)}")

        if len(tfr_cycles) == 0:
            print("  No gait cycles survived -- skipping subject.")
            continue

        # ERSP

        tfr_stack = np.stack(tfr_cycles)   # (n_cycles, n_ch, n_freqs, N_POINTS)

        # Per-cycle log ratio to standing baseline, then average across cycles.
        # This formulation treats each cycle as an independent observation.
        # Averaging after log conversion avoids upweighting high-power cycles.
        # Makeig et al. 1993 Electroencephalogr Clin Neurophysiol;
        # Delorme & Makeig 2004 J Neurosci Methods
        ersp_per_cycle = 10 * np.log10(
            tfr_stack / baseline_power[np.newaxis, :, :, np.newaxis]
        )   # (n_cycles, n_ch, n_freqs, N_POINTS)

        ersp_avg = ersp_per_cycle.mean(axis=0)   # (n_ch, n_freqs, N_POINTS)

        n_nan = np.sum(np.isnan(ersp_avg))
        print(f"\n  ERSP shape : {ersp_avg.shape}")
        print(f"  NaN count  : {n_nan}")
        print(f"  Range      : {ersp_avg.min():.2f} / {ersp_avg.max():.2f} dB")

        # --- Linear ROI weights ---
        # Compute once per subject from the ICA-cleaned raw channel positions.
        # Saved for reproducibility and topography plotting.
        # Weights are used at analysis time in multiverse_pipeline.py and
        # prepana06_plotbetagait.py — the full per-channel ERSP is preserved.
        _info = concat_raw.info
        try:
            weights = linear_roi_weights(_info, center_ch="Cz")
            np.save(out_weights, weights)

            plot_weight_topography(
                weights, _info, subject,
                out_path  = ROI_TOPO_DIR / f"sub-{subject}_roi_weights_topo.png",
                center_ch = "Cz",
            )
            print(f"  ROI weights saved  max_ch={_info.ch_names[weights.argmax()]}  "
                  f"max_w={weights.max():.4f}")
        except Exception as e:
            print(f"  [WARN] ROI weights failed: {e}")
            weights = None

        if weights is not None:
            roi_mean_weighted = float(apply_linear_roi(ersp_avg, weights).mean())
        else:
            roi_mean_weighted = float("nan")
        print(f"  Linear ROI mean: {roi_mean_weighted:+.2f} dB  "
              f"(center=Cz)")

        # --- Phase ERSP (double stance vs swing), event-anchored ---

        # Because every cycle was warped above so that its own LTO/LHS/RTO
        # land exactly at the group-median anchors (A_lto/A_lhs/A_rto),
        # phase windows are now fixed index ranges on the common 101-point
        # grid -- the same ranges for every cycle and every subject.
        # Double support occurs twice per gait cycle (initial DS right
        # after RHS, ending at LTO; terminal DS right after LHS, ending
        # at RTO); swing likewise occurs twice (right swing LTO->LHS,
        # left swing RTO->RHS). See: Perry & Burnfield, Gait Analysis:
        # Normal and Pathological Function, 2nd ed., Ch. 1.
        double_stance_idx, swing_idx = phase_split_indices(
            (A_lto, A_lhs, A_rto), n_points=N_POINTS
        )

        print(f"  Phase windows (event-anchored, % of gait cycle): "
              f"DS1=[0,{A_lto:.1f}]  SW1=[{A_lto:.1f},{A_lhs:.1f}]  "
              f"DS2=[{A_lhs:.1f},{A_rto:.1f}]  SW2=[{A_rto:.1f},100]")

        # For each cycle, average ERSP over the double-stance samples and
        # over the swing samples, then average across cycles.
        # Shape of each: (n_ch, n_freqs)
        double_stance_ersp_per_cycle = ersp_per_cycle[:, :, :, double_stance_idx].mean(axis=-1)
        swing_ersp_per_cycle         = ersp_per_cycle[:, :, :, swing_idx].mean(axis=-1)

        ersp_double_stance = double_stance_ersp_per_cycle.mean(axis=0)  # (n_ch, n_freqs)
        ersp_swing         = swing_ersp_per_cycle.mean(axis=0)          # (n_ch, n_freqs)

        _smx = [stand_ch_names.index(c) for c in ['Cz', 'C3', 'C4'] if c in stand_ch_names]
        print(f"  Double stance ERSP (segment, sensorimotor mean): "
              f"{ersp_avg[_smx][:, :, double_stance_idx].mean():+.2f} dB")
        print(f"  Swing         ERSP (segment, sensorimotor mean): "
              f"{ersp_avg[_smx][:, :, swing_idx].mean():+.2f} dB")

        np.save(out_ds, ersp_double_stance)
        np.save(out_sw, ersp_swing)
        print(f"  Saved -> {out_ds.name}  shape={ersp_double_stance.shape}")
        print(f"  Saved -> {out_sw.name}  shape={ersp_swing.shape}")

        np.save(out_beta, ersp_avg)
        print(f"\n  Saved -> {out_beta.name}")

        # --- QC: ERSP ---
        n_nan         = int(np.isnan(ersp_avg).sum())
        ersp_range    = float(ersp_avg.max() - ersp_avg.min())
        roi_mean      = roi_mean_weighted
        n_cycles_used = len(tfr_cycles)

        if n_nan > 0 or n_cycles_used < 20:
            ersp_flag = "fail"
        elif n_cycles_used < 50 or ersp_range < 1.0:
            ersp_flag = "warn"
        else:
            ersp_flag = "pass"

        # Baseline sanity check: mean beta power across channels should be
        # within physiological range for preprocessed EEG (1e-13 to 1e-9 V^2).
        # Values outside this range indicate a corrupted baseline.
        baseline_mean = float(baseline_power.mean())
        baseline_ok   = 1e-13 < baseline_mean < 1e-9
        if not baseline_ok:
            print(f"  [WARN] sub-{subject}: baseline mean {baseline_mean:.3e} "
                  f"outside physiological range [1e-13, 1e-9] V^2")
            ersp_flag = "warn"

        log_qc(
            qc_dir  = QC_DIR,
            subject = subject,
            stage   = "ersp",
            flag    = ersp_flag,
            metrics = {
                "n_cycles_used":   n_cycles_used,
                "n_nan":           n_nan,
                "ersp_range_db":   round(ersp_range, 2),
                "roi_mean_db":     round(roi_mean, 2) if not np.isnan(roi_mean) else None,
                "double_stance_roi_db": round(float(apply_linear_roi(ersp_double_stance, weights).mean()), 2) if weights is not None else None,
                "swing_roi_db":         round(float(apply_linear_roi(ersp_swing,         weights).mean()), 2) if weights is not None else None,
                "baseline_mean_v2": float(round(baseline_mean, 20)),
                "baseline_ok":     baseline_ok,
            },
        )
        print(f"  QC ersp: {ersp_flag}  "
              f"cycles={n_cycles_used}  nan={n_nan}  "
              f"range={ersp_range:.2f}dB  roi_mean={roi_mean:.2f}dB")

    except FileNotFoundError as e:
        print(f"\n  [SKIP] sub-{subject}: file not found -- {e}")
        continue
    except Exception as e:
        print(f"\n  [ERROR] sub-{subject}: unexpected error -- {e}")
        import traceback
        traceback.print_exc()
        continue

print("\nDone")
print(f"\nDone. Processed {len(LOOP_SUBJECTS)} subject(s): {LOOP_SUBJECTS}")
