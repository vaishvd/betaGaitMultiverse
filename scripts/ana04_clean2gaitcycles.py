"""
ana04_clean2gaitcycles.py
=========================
Extract gait-cycle EEG segments from ICA-cleaned data at original sfreq.

# Gait segments saved at original sfreq for TFR computation.
# Time-normalisation is applied AFTER TFR on the power time series,
# not on the raw voltage signal. This preserves instantaneous frequency
# content during wavelet convolution.
# See: Seeber et al. 2015 J Neurosci; Daly et al. 2011 J Neural Eng

Input
-----
d01_gaitevents/  sub-{sub}_cycles.tsv        (rhs_start_s, rhs_end_s in s)
d03_clean/       sub-{sub}_desc-icaClean_raw.fif

Output
------
d04_gaitepochs/  sub-{sub}_gait_segments.npy   object array (n_cycles,), each (n_ch, n_samp_i)
                 sub-{sub}_gait_sfreq.npy       scalar float, recording sfreq
                 sub-{sub}_cycles_kept.tsv      TSV of kept cycle metadata
"""

import numpy as np
import pandas as pd
import mne

from src.paths import get_dataset_dirs

DATASET  = "stepup"
SUBJECTS = ["S1"]

AMP_THRESH = 350e-6  # V

dirs = get_dataset_dirs(DATASET)

GAIT_EVENTS_DIR = dirs["gait_events"]
CLEAN_DIR       = dirs["clean"]
GAITEPOCH_DIR   = dirs["gaitepochs"]


for subject in SUBJECTS:

    print(f"GAIT SEGMENTATION: sub-{subject}")

    raw_path = CLEAN_DIR / f"sub-{subject}_desc-icaClean_raw.fif"

    if not raw_path.exists():
        print(f"  Clean EEG not found, skipping: {raw_path.name}")
        continue

    raw   = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    eeg_dur = raw.times[-1]

    print(f"  EEG sfreq    : {sfreq:.0f} Hz")
    print(f"  EEG duration : {eeg_dur:.2f} s  ({raw.n_times} samples)")
    print(f"  EEG channels : {len(raw.ch_names)}")

    all_data  = raw.get_data()
    nan_frac  = np.mean(np.isnan(all_data))
    nan_by_ch = np.mean(np.isnan(all_data), axis=1)
    bad_chs   = [raw.ch_names[i] for i, f in enumerate(nan_by_ch) if f > 0.01]

    if bad_chs:
        print(f"  Channels with >1% NaN ({len(bad_chs)}): {bad_chs}")
        print(f"  These channels will be excluded from segments.")

    cycles_path = GAIT_EVENTS_DIR / f"sub-{subject}_cycles.tsv"
    cycles_df   = pd.read_csv(cycles_path, sep="\t")

    if len(cycles_df) == 0:
        print("  ERROR: cycles.tsv is empty -- skipping subject.")
        continue

    gait_t0 = cycles_df["rhs_start_s"].min()
    gait_t1 = cycles_df["rhs_end_s"].max()

    print(f"\n  Gait cycles loaded : {len(cycles_df)}")
    print(f"  Gait time range    : {gait_t0:.2f} - {gait_t1:.2f} s")
    print(f"  Mean cycle duration: {cycles_df['duration_s'].mean():.3f} s")

    if gait_t0 < 0 or gait_t1 > eeg_dur + 1.0:
        print(f"\n  WARNING: gait cycles [{gait_t0:.1f}, {gait_t1:.1f}] s "
              f"extend beyond EEG [{0:.1f}, {eeg_dur:.1f}] s -- "
              f"check synchronisation.")

    gait_segments = []
    kept_meta     = []
    skipped_oob   = 0
    skipped_bad   = 0
    rejection_log = []

    for _, row in cycles_df.iterrows():

        t_start = float(row["rhs_start_s"])
        t_end   = float(row["rhs_end_s"])

        i_start = int(round(t_start * sfreq))
        i_end   = int(round(t_end   * sfreq))

        if i_start < 0 or i_end > raw.n_times or i_end <= i_start:
            if len(rejection_log) < 5:
                rejection_log.append(
                    f"OOB: cycle {_} t=[{t_start:.3f},{t_end:.3f}]s "
                    f"-> samples [{i_start},{i_end}] (n_times={raw.n_times})"
                )
            skipped_oob += 1
            continue

        data = raw.get_data(start=i_start, stop=i_end)   # (n_ch, n_samp_i)

        if not np.isfinite(data).all():
            if len(rejection_log) < 5:
                rejection_log.append(f"NaN/Inf: cycle {_} t=[{t_start:.3f},{t_end:.3f}]s")
            skipped_bad += 1
            continue

        peak_amp = np.max(np.abs(data))

        if peak_amp > AMP_THRESH:
            skipped_bad += 1
            if len(rejection_log) < 5:
                rejection_log.append(
                    f"Amplitude: cycle {_} peak={peak_amp*1e6:.1f} uV"
                )
            continue

        gait_segments.append(data)
        kept_meta.append(row)

    n_extracted = len(gait_segments)
    print(f"\n  Extracted  : {n_extracted} / {len(cycles_df)} cycles")
    print(f"  Skipped OOB: {skipped_oob}")
    print(f"  Skipped bad: {skipped_bad}")

    if rejection_log:
        print("  First rejected cycles:")
        for reason in rejection_log:
            print(f"    {reason}")

    if n_extracted == 0:
        print("  No segments extracted -- skipping subject.")
        continue

    meta_df = pd.DataFrame(kept_meta).reset_index(drop=True)

    n_samp_min = min(s.shape[1] for s in gait_segments)
    n_samp_max = max(s.shape[1] for s in gait_segments)
    print(f"  Segment lengths: min={n_samp_min}  max={n_samp_max} samples")

    segments_arr = np.empty(n_extracted, dtype=object)
    for i, seg in enumerate(gait_segments):
        segments_arr[i] = seg

    out_seg   = GAITEPOCH_DIR / f"sub-{subject}_gait_segments.npy"
    out_sfreq = GAITEPOCH_DIR / f"sub-{subject}_gait_sfreq.npy"
    out_meta  = GAITEPOCH_DIR / f"sub-{subject}_cycles_kept.tsv"

    np.save(out_seg,   segments_arr, allow_pickle=True)
    np.save(out_sfreq, np.float64(sfreq))
    meta_df.to_csv(out_meta, sep="\t", index=False)

    print(f"\n  Saved -> {out_seg.name}")
    print(f"  Saved -> {out_sfreq.name}")
    print(f"  Saved -> {out_meta.name}")

print("\nGAIT SEGMENTATION COMPLETE")
