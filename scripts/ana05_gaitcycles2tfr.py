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
    sub-{sub}_desc-icaClean_raw.fif   (channel reference + ICA)
    sub-{sub}_ica-clean.fif

d04_gaitepochs/
    sub-{sub}_gait_segments.npy       object array (n_cycles,), each (n_ch, n_samp_i)
    sub-{sub}_gait_sfreq.npy          scalar float

d00_raw/
    sub-{sub}/eeg/sub-{sub}_task-STAND.vhdr

Output
------
d05_ersp/
    sub-{sub}_ersp_beta.npy           shape (n_ch, n_freqs, N_POINTS)
"""

import numpy as np
import mne

from src.paths import get_dataset_dirs
from src.preprocessing import drop_invalid_channels

DATASET  = "stepup"
SUBJECTS = ["S1"]

TASK_STAND = "STAND"

TARGET_SFREQ       = 250
BAD_CHAN_THRESHOLD = 3.0

FREQS    = np.arange(13, 31, dtype=float)   # beta band 13-30 Hz
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

L_FREQ    = 1.0
LINE_FREQ = 50

AMP_THRESH_BASELINE = 350e-6
AMP_THRESH_CYCLE    = 350e-6

dirs = get_dataset_dirs(DATASET)

RAW_DIR       = dirs["raw"]
CLEAN_DIR     = dirs["clean"]
GAITEPOCH_DIR = dirs["gaitepochs"]
ERSP_DIR      = dirs["ersp"]


for subject in SUBJECTS:

    print(f"ERSP: sub-{subject}")

    # CHANNEL REFERENCE FROM ICA-CLEAN RAW

    clean_fif     = CLEAN_DIR / f"sub-{subject}_desc-icaClean_raw.fif"
    raw_ref       = mne.io.read_raw_fif(clean_fif, preload=False, verbose=False)
    ch_names_ref  = list(raw_ref.ch_names)
    print(f"  Reference channels from clean raw: {len(ch_names_ref)}")

    # Load Gait Segments and Check Cleaned Data Quality

    seg_path   = GAITEPOCH_DIR / f"sub-{subject}_gait_segments.npy"
    sfreq_path = GAITEPOCH_DIR / f"sub-{subject}_gait_sfreq.npy"

    if not seg_path.exists():
        print(f"  Gait segments not found: {seg_path.name} -- skipping.")
        continue

    gait_segments = np.load(seg_path, allow_pickle=True)
    gait_sfreq    = float(np.load(sfreq_path))

    print(f"  Gait segments: {len(gait_segments)}  sfreq={gait_sfreq:.0f} Hz")

    # Standing baseline  

    stand_file = (
        RAW_DIR
        / f"sub-{subject}"
        / "eeg"
        / f"sub-{subject}_task-{TASK_STAND}.vhdr"
    )

    raw_stand = mne.io.read_raw_brainvision(stand_file, preload=True, verbose=False)
    raw_stand.pick(ch_names_ref, verbose=False)
    raw_stand = drop_invalid_channels(raw_stand)

    if raw_stand.info["sfreq"] > TARGET_SFREQ:
        raw_stand.resample(TARGET_SFREQ)

    raw_stand.filter(l_freq=L_FREQ, h_freq=60, fir_design="firwin")
    raw_stand.notch_filter(freqs=LINE_FREQ)

    data = raw_stand.get_data()
    ptp  = np.ptp(data, axis=1)
    z    = (ptp - np.mean(ptp)) / np.std(ptp)
    bads = [raw_stand.ch_names[i] for i in np.where(np.abs(z) > BAD_CHAN_THRESHOLD)[0]]
    print(f"  Bad channels detected: {bads}")

    raw_stand.info["bads"] = bads
    if len(bads) > 0:
        raw_stand.interpolate_bads(reset_bads=True)

    raw_stand.set_eeg_reference("average", projection=False)

    ica = mne.preprocessing.read_ica(
        CLEAN_DIR / f"sub-{subject}_ica-clean.fif", verbose=False
    )
    ica.apply(raw_stand, verbose=False)

    stand_sfreq  = raw_stand.info["sfreq"]
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

    stand_tfr_stack = np.stack(stand_tfr_list)   # (n_kept, n_ch, n_freqs, n_time_cropped)
    baseline_power  = stand_tfr_stack.mean(axis=(0, 3))   # (n_ch, n_freqs)

    print(f"  Baseline shape: {baseline_power.shape}  "
          f"range=[{baseline_power.min():.4e}, {baseline_power.max():.4e}]")

    # Channel index map: gait segments (ch_names_ref order) -> standing (stand_ch_names order)
    ref_idx = [ch_names_ref.index(ch) for ch in stand_ch_names if ch in ch_names_ref]
    if len(ref_idx) < len(stand_ch_names):
        dropped = [ch for ch in stand_ch_names if ch not in ch_names_ref]
        print(f"  WARNING: {len(dropped)} standing channels not in reference: {dropped}")

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

    ROI_CHANNELS = ["Cz", "C3", "C4", "FC1", "FC2", "FCz", "CP1", "CP2", "CPz"]
    for ch in ROI_CHANNELS:
        if ch in stand_ch_names:
            i = stand_ch_names.index(ch)
            print(f"  {ch:>8s}  mean={ersp_avg[i].mean():+.2f}  "
                  f"min={ersp_avg[i].min():+.2f}  max={ersp_avg[i].max():+.2f} dB")
        else:
            print(f"  {ch:>8s}  NOT FOUND in channel list")

    global_mean = ersp_avg.mean()
    print(f"  Global ERSP mean : {global_mean:+.2f} dB")
 
    out = ERSP_DIR / f"sub-{subject}_ersp_beta.npy"
    np.save(out, ersp_avg)
    print(f"\n  Saved -> {out.name}")

print("\nDone")
