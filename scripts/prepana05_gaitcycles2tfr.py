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

import numpy as np
import pandas as pd
import mne

from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS
from src.qc import log_qc
from src.spatial_filter import linear_roi_weights, apply_linear_roi, plot_weight_topography

FREQS    = np.arange(8, 41, dtype=float)    # 8-40 Hz (alpha-beta range)
N_CYCLES = FREQS / 2.0                      # frequency-dependent wavelet width
N_POINTS = 101  # 0-100% gait cycle in 1% steps; below native resolution (~248 samples at 250 Hz), avoids upsampling artefacts
EDGE_CROP = 0.05                            # fraction trimmed at each edge post-TFR
DOUBLE_STANCE_WINDOWS = [(0, 20), (50, 70)]   # gait cycle % for segment QC print
SWING_WINDOWS         = [(20, 50), (70, 100)]

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


for subject in SUBJECTS:
    try:
        print(f"ERSP: sub-{subject}")

        # Load ICA-cleaned concatenated raw; preload=True so STAND segment can be extracted
        clean_fif    = CLEAN_DIR / f"sub-{subject}_desc-icaClean_concat_raw.fif"
        concat_raw   = mne.io.read_raw_fif(clean_fif, preload=True, verbose=False)
        ch_names_ref = list(concat_raw.ch_names)
        print(f"  Reference channels from clean raw: {len(ch_names_ref)}")

        # Load Gait Segments

        seg_path   = GAITEPOCH_DIR / f"sub-{subject}_gait_segments.npy"
        sfreq_path = GAITEPOCH_DIR / f"sub-{subject}_gait_sfreq.npy"

        if not seg_path.exists():
            print(f"  Gait segments not found: {seg_path.name} -- skipping.")
            continue

        gait_segments = np.load(seg_path, allow_pickle=True)
        gait_sfreq    = float(np.load(sfreq_path))

        print(f"  Gait segments: {len(gait_segments)}  sfreq={gait_sfreq:.0f} Hz")

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

        stand_sfreq    = raw_stand.info["sfreq"]
        stand_ch_names = list(raw_stand.ch_names)

        events = mne.make_fixed_length_events(raw_stand, duration=2.0)
        stand_epochs = mne.Epochs(
            raw_stand, events,
            tmin=0, tmax=2.0,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            verbose=False,
        )
        stand_epochs.drop_bad(reject=dict(eeg=AMP_THRESH_BASELINE))
        print(f"  Standing epochs kept: {len(stand_epochs)}")

        stand_tfr_list = []

        for epoch in stand_epochs.get_data():   # epoch: (n_ch, n_time)
            if np.max(np.abs(epoch)) > AMP_THRESH_BASELINE:
                continue
            tfr = mne.time_frequency.tfr_array_morlet(
                epoch[np.newaxis],
                sfreq=stand_sfreq,
                freqs=FREQS,
                n_cycles=N_CYCLES,
                output="power",
                zero_mean=True,
                verbose=False,
            )[0]   # (n_ch, n_freqs, n_time)

            crop = int(EDGE_CROP * tfr.shape[-1])
            if crop > 0:
                tfr = tfr[..., crop:-crop]

            stand_tfr_list.append(tfr)

        print(f"  Standing epochs accepted for baseline: {len(stand_tfr_list)} / "
              f"{len(stand_epochs)}")

        if len(stand_tfr_list) == 0:
            raise RuntimeError(
                f"sub-{subject}: all standing epochs rejected -- "
                f"cannot compute baseline"
            )

        stand_tfr_stack = np.stack(stand_tfr_list)   # (n_kept, n_ch, n_freqs, n_time_cropped)
        baseline_power  = stand_tfr_stack.mean(axis=(0, 3))   # (n_ch, n_freqs)
        assert baseline_power.shape == (stand_tfr_stack.shape[1], stand_tfr_stack.shape[2]), \
            f"Baseline shape {baseline_power.shape} does not match expected (n_ch, n_freqs)"

        print(f"  Baseline shape: {baseline_power.shape}  "
              f"range=[{baseline_power.min():.4e}, {baseline_power.max():.4e}]")

        # Gait segments and standing share the same source file; channels are identical
        ref_idx = [ch_names_ref.index(ch) for ch in stand_ch_names if ch in ch_names_ref]

        # Per cycle TFR

        tfr_cycles = []
        n_rej = 0

        for seg in gait_segments:   # seg: (n_ch_ref, n_samp_i)

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

            # Resample power time axis to N_POINTS using linear interpolation
            n_t   = power.shape[-1]
            x_old = np.linspace(0, 1, n_t)
            x_new = np.linspace(0, 1, N_POINTS)
            resampled = np.stack([
                np.stack([np.interp(x_new, x_old, power[ch, f])
                          for f in range(power.shape[1])])
                for ch in range(power.shape[0])
            ])   # (n_ch, n_freqs, N_POINTS)

            tfr_cycles.append(resampled)

        print(f"  Gait cycles rejected by amplitude: {n_rej}")
        print(f"  Gait cycles accepted             : {len(tfr_cycles)}")

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
            np.save(ERSP_DIR / f"sub-{subject}_roi_weights.npy", weights)

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

        # --- Phase ERSP (stance vs swing) ---

        # Per-cycle stance/swing split based on RTO timing.
        # For a right-foot cycle (RHS-to-RHS), stance ends at RTO.
        # Split point is computed as fraction of cycle duration, then
        # mapped to the 101-point normalized time axis.
        # This respects intra-individual variability in stance/swing ratio
        # rather than assuming a fixed 60/40 split.
        # See: Kline et al. 2022 J Neurophysiol; Handford et al. 2022 Gait Posture
        cycles_meta = pd.read_csv(
            GAITEPOCH_DIR / f"sub-{subject}_cycles_kept.tsv", sep="\t"
        )
        rto_fracs = (
            (cycles_meta["rto_s"] - cycles_meta["rhs_start_s"]) /
            (cycles_meta["rhs_end_s"] - cycles_meta["rhs_start_s"])
        ).values  # shape: (n_cycles,)

        # Clip to valid range -- rto_frac should be in (0.3, 0.8) for normal gait
        rto_fracs = np.clip(rto_fracs, 0.3, 0.8)

        # Convert fraction to index in 101-point axis
        rto_indices = np.round(rto_fracs * (N_POINTS - 1)).astype(int)  # per cycle

        print(f"  RTO fraction: mean={rto_fracs.mean():.3f}  "
              f"std={rto_fracs.std():.3f}  "
              f"range=[{rto_fracs.min():.3f}, {rto_fracs.max():.3f}]")

        # Guard: cycles_kept.tsv must match ersp_per_cycle cycle count.
        # Both ana04 and ana05 apply AMP_THRESH=350e-6 to the same segments,
        # so counts should always match. Raise early if they diverge.
        if len(cycles_meta) != ersp_per_cycle.shape[0]:
            raise RuntimeError(
                f"cycles_kept.tsv has {len(cycles_meta)} rows but ersp_per_cycle has "
                f"{ersp_per_cycle.shape[0]} cycles -- amplitude rejection mismatch."
            )

        # For each cycle, average ERSP over stance samples (0 to rto_idx)
        # and swing samples (rto_idx to N_POINTS).
        # Average across cycles after phase separation.
        # Shape of each: (n_ch, n_freqs)
        double_stance_ersp_per_cycle = np.stack([
            ersp_per_cycle[k, :, :, :rto_indices[k]].mean(axis=-1)
            for k in range(len(rto_indices))
        ])  # (n_cycles, n_ch, n_freqs)

        swing_ersp_per_cycle = np.stack([
            ersp_per_cycle[k, :, :, rto_indices[k]:].mean(axis=-1)
            for k in range(len(rto_indices))
        ])  # (n_cycles, n_ch, n_freqs)

        ersp_double_stance = double_stance_ersp_per_cycle.mean(axis=0)  # (n_ch, n_freqs)
        ersp_swing         = swing_ersp_per_cycle.mean(axis=0)          # (n_ch, n_freqs)

        _pi     = lambda pct: int(round(pct / 100 * (N_POINTS - 1)))
        _ds_idx = np.concatenate([np.arange(_pi(s), _pi(e)) for s, e in DOUBLE_STANCE_WINDOWS])
        _sw_idx = np.concatenate([np.arange(_pi(s), _pi(e)) for s, e in SWING_WINDOWS])
        _smx    = [stand_ch_names.index(c) for c in ['Cz', 'C3', 'C4'] if c in stand_ch_names]
        print(f"  Double stance ERSP (segment, sensorimotor mean): "
              f"{ersp_avg[_smx][:, :, _ds_idx].mean():+.2f} dB")
        print(f"  Swing         ERSP (segment, sensorimotor mean): "
              f"{ersp_avg[_smx][:, :, _sw_idx].mean():+.2f} dB")

        np.save(ERSP_DIR / f"sub-{subject}_ersp_double_stance.npy", ersp_double_stance)
        np.save(ERSP_DIR / f"sub-{subject}_ersp_swing.npy",         ersp_swing)
        print(f"  Saved -> sub-{subject}_ersp_double_stance.npy  shape={ersp_double_stance.shape}")
        print(f"  Saved -> sub-{subject}_ersp_swing.npy           shape={ersp_swing.shape}")

        pd.DataFrame({
            "rto_frac": rto_fracs,
            "rto_idx":  rto_indices
        }).to_csv(ERSP_DIR / f"sub-{subject}_rto_fracs.csv", index=False)

        out = ERSP_DIR / f"sub-{subject}_ersp_beta.npy"
        np.save(out, ersp_avg)
        print(f"\n  Saved -> {out.name}")

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
print(f"\nDone. Processed {len(SUBJECTS)} subject(s): {SUBJECTS}")
