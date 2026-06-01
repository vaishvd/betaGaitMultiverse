"""
ana04_clean2gaitcycles.py
=========================
Align ICA-cleaned continuous EEG to gait cycles and time-normalise
each cycle to N_POINTS samples (0 % – 100 % of stride).

Input
-----
d01_gaitevents/  sub-{sub}_cycles.tsv        (rhs_start_s, rhs_end_s in s)
d03_clean/       sub-{sub}_desc-icaClean_raw.fif

Output
------
d04_gaitepochs/  sub-{sub}_gait_epo.fif      (n_cycles × n_ch × N_POINTS)
                 sub-{sub}_gait_epo_qc.png
"""

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import resample
from src.paths import get_dataset_dirs

DATASET  = "stepup"
SUBJECTS = ["S1"]

# Time-normalised samples per gait cycle (0 % – 100 % of stride)
N_POINTS = 512

dirs = get_dataset_dirs(DATASET)

GAIT_EVENTS_DIR = dirs["gait_events"]
CLEAN_DIR       = dirs["clean"]
GAITEPOCH_DIR   = dirs["gaitepochs"]


for subject in SUBJECTS:

    print(f"\n{'='*60}")
    print(f"GAIT EPOCHING: sub-{subject}")
    print(f"{'='*60}")

    # Load ICA-cleaned raw 

    raw_path = CLEAN_DIR / f"sub-{subject}_desc-icaClean_raw.fif"

    if not raw_path.exists():
        print(f"  Clean EEG not found, skipping: {raw_path.name}")
        continue

    raw  = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    sfreq    = raw.info["sfreq"]
    eeg_dur  = raw.times[-1]

    print(f"  EEG sfreq    : {sfreq:.0f} Hz")
    print(f"  EEG duration : {eeg_dur:.2f} s  ({raw.n_times} samples)")
    print(f"  EEG channels : {len(raw.ch_names)}")


    all_data  = raw.get_data()
    nan_frac  = np.mean(np.isnan(all_data))
    nan_by_ch = np.mean(np.isnan(all_data), axis=1)
    bad_chs   = [raw.ch_names[i] for i, f in enumerate(nan_by_ch) if f > 0.01]

    print(f"\n  NaN fraction (whole recording): {nan_frac*100:.1f}%")

    if nan_frac > 0.50:
        print(f"  ERROR: >{50}% of EEG data is NaN — the ICA-cleaned file appears "
              f"corrupt. Re-run the cleaning step (ana03_ica2clean) before epoching.")
        continue

    if bad_chs:
        print(f"  Channels with >1% NaN ({len(bad_chs)}): {bad_chs}")
        print(f"  These channels will be excluded from epochs.")

    #  Load gait cycles 

    cycles_path = GAIT_EVENTS_DIR / f"sub-{subject}_cycles.tsv"
    cycles_df   = pd.read_csv(cycles_path, sep="\t")

    gait_t0  = cycles_df["rhs_start_s"].min()
    gait_t1  = cycles_df["rhs_end_s"].max()

    print(f"\n  Gait cycles loaded : {len(cycles_df)}")
    print(f"  Gait time range    : {gait_t0:.2f} – {gait_t1:.2f} s")
    print(f"  Mean cycle duration: {cycles_df['duration_s'].mean():.3f} s")

    #  Alignment check

    if gait_t0 < 0 or gait_t1 > eeg_dur + 1.0:
        print(f"\n  WARNING: gait cycles [{gait_t0:.1f}, {gait_t1:.1f}] s "
              f"extend beyond EEG [{0:.1f}, {eeg_dur:.1f}] s — "
              f"check synchronisation.")
    else:
        print(f"  Alignment OK: gait cycles are within EEG recording.")

    # Extract and time-normalise 

    gait_epochs  = []
    kept_meta    = []
    skipped_oob  = 0
    skipped_bad  = 0

    for _, row in cycles_df.iterrows():

        t_start = float(row["rhs_start_s"])   # seconds 
        t_end   = float(row["rhs_end_s"])

        i_start = int(round(t_start * sfreq))
        i_end   = int(round(t_end   * sfreq))

        if i_start < 0 or i_end > raw.n_times or i_end <= i_start:
            skipped_oob += 1
            continue

        data = raw.get_data(start=i_start, stop=i_end)   # (n_ch, n_samp)

        if not np.isfinite(data).all():
            skipped_bad += 1
            continue

        gait_epochs.append(resample(data, N_POINTS, axis=1))
        kept_meta.append(row)

    n_extracted = len(gait_epochs)
    print(f"\n  Extracted  : {n_extracted} / {len(cycles_df)} cycles")
    print(f"  Skipped OOB: {skipped_oob}")
    print(f"  Skipped bad: {skipped_bad}")

    if n_extracted == 0:
        print("  No epochs extracted — skipping subject.")
        continue

    gait_epochs = np.stack(gait_epochs)          # (n_epochs, n_ch, N_POINTS)
    meta_df     = pd.DataFrame(kept_meta).reset_index(drop=True)

    print(f"  Array shape : {gait_epochs.shape}  "
          f"(epochs × channels × normalised_samples)")

    # Amplitude sanity check

    amp_uv = gait_epochs * 1e6
    print(f"\n  Amplitude (µV):")
    print(f"    range : [{amp_uv.min():.1f}, {amp_uv.max():.1f}]")
    print(f"    mean abs: {np.abs(amp_uv).mean():.2f}")

    if np.abs(amp_uv).max() > 500:
        print("  WARNING: extreme amplitudes >500 µV — inspect for artefacts.")

    #  Build MNE EpochsArray 
    # sfreq = N_POINTS so time axis reads 0.0 – 1.0 = 0 % – 100 % of stride

    info = raw.info.copy()
    with info._unlock():
        info["sfreq"] = float(N_POINTS)

    events = np.column_stack([
        np.arange(n_extracted),
        np.zeros(n_extracted, dtype=int),
        np.ones(n_extracted,  dtype=int),
    ])

    epochs = mne.EpochsArray(
        gait_epochs,
        info,
        events=events,
        event_id={"gait_cycle": 1},
        tmin=0.0,
        metadata=meta_df[["cycle_id", "duration_s", "lto_frac",
                           "rhs_start_s", "rhs_end_s"]],
        verbose=False,
    )

    # Save 

    out = GAITEPOCH_DIR / f"sub-{subject}_gait_epo.fif"
    epochs.save(out, overwrite=True, verbose=False)
    print(f"\n  Saved → {out.name}")

    # QC plot 
    # 6 channels spread across scalp, mean ± SEM across all cycles

    pct     = np.linspace(0, 100, N_POINTS)
    ch_idx  = np.linspace(0, len(raw.ch_names) - 1, 6, dtype=int)
    ch_names = [raw.ch_names[i] for i in ch_idx]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flat

    for ax, ch, ci in zip(axes, ch_names, ch_idx):
        traces = gait_epochs[:, ci, :] * 1e6   # µV, (n_epochs, N_POINTS)
        mean   = traces.mean(axis=0)
        sem    = traces.std(axis=0) / np.sqrt(n_extracted)

        ax.fill_between(pct, mean - sem, mean + sem,
                        color="steelblue", alpha=0.3)
        ax.plot(pct, mean, color="steelblue", lw=1.5)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(ch, fontsize=9)
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("µV")

    fig.suptitle(
        f"sub-{subject}  |  time-normalised gait epochs  "
        f"(n={n_extracted}, mean ± SEM)",
        fontsize=10,
    )
    plt.tight_layout()

    qc_path = GAITEPOCH_DIR / f"sub-{subject}_gait_epo_qc.png"
    fig.savefig(qc_path, dpi=150)
    plt.close(fig)
    print(f"  QC plot → {qc_path.name}")


print("\nGAIT EPOCHING COMPLETE")
