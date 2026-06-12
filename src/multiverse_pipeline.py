"""
Single-subject multiverse pipeline.

Runs one complete analysis branch for one subject under one set of
decision-node choices. Returns a results dict containing the primary
outcome (t-statistic for stance vs swing beta difference) and
supporting metrics.

Hybrid persistence strategy:
  - ICA output is written to a branch-specific directory so the
    expensive ICA step is not repeated if downstream stages fail.
  - Everything after ICA is computed in memory and returned as a dict.

Called by COMET universe scripts via run_subject_multiverse().
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path

from src.paths import get_dataset_dirs
from src.preprocessing import drop_invalid_channels, drop_invalid_eeg_channels
from src.ica_utils import run_ica, label_and_mark_ica
from src.nodes.asr_node import apply_asr_node
from src.config import DIR_MULTIVERSE_BRANCHES

TARGET_SFREQ = 250
EPOCH_DUR    = 2.0
N_COMPONENTS = 0.99
RANDOM_STATE = 42
AMP_THRESH   = 350e-6
FREQS        = np.arange(13, 31, dtype=float)
N_CYCLES_WAV = FREQS / 2.0
N_POINTS     = 101
EDGE_CROP    = 0.05
ROI_CHANNEL  = "Cz"
LINE_FREQ    = 50.0


def _branch_dir(base_dir: Path, decisions: dict) -> Path:
    """
    Return a branch-specific subdirectory path derived from decisions.

    The directory name is a deterministic string built from sorted
    decision key-value pairs, e.g.:
        baseline_type-standing_brain_thresh-0.7_highpass_hz-0.1_...
    This ensures the same decisions always map to the same directory
    and different decisions never collide.
    """
    parts = "_".join(
        f"{k}-{v}" for k, v in sorted(decisions.items())
    )
    branch = Path(base_dir) / parts
    branch.mkdir(parents=True, exist_ok=True)
    return branch


def run_subject_multiverse(
    subject: str,
    dataset: str,
    decisions: dict,
) -> dict:
    """
    Run one analysis branch for one subject.

    Parameters
    ----------
    subject   : str   Subject id, e.g. 'S1'
    dataset   : str   Dataset key, e.g. 'stepup'
    decisions : dict  Keys: use_asr, brain_thresh, highpass_hz,
                      lowpass_hz, baseline_type

    Returns
    -------
    dict with keys:
        subject, t_stat, t_pval, beta_stance_mean, beta_swing_mean,
        n_cycles, n_brain_ics, baseline_ok, + all decision keys
    """
    from scipy.stats import ttest_rel
    from autoreject import AutoReject

    dirs       = get_dataset_dirs(dataset)
    raw_dir    = dirs["raw"]
    event_dir  = dirs["gait_events"]
    branch_dir = _branch_dir(DIR_MULTIVERSE_BRANCHES / subject, decisions)

    # ------------------------------------------------------------------ #
    # Stage 1 — Load and concatenate
    # ------------------------------------------------------------------ #
    stand_vhdr = raw_dir / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-STAND.vhdr"
    walk_vhdr  = raw_dir / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-CS.vhdr"
    for p in (stand_vhdr, walk_vhdr):
        if not p.exists():
            raise FileNotFoundError(str(p))

    montage = mne.channels.make_standard_montage("standard_1005")

    def _load_eeg(path):
        r = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        r.pick_types(eeg=True)
        r = drop_invalid_channels(r)
        r.pick("eeg")
        r.set_montage(montage, on_missing="ignore")
        return r

    raw_stand = _load_eeg(stand_vhdr)
    raw_walk  = _load_eeg(walk_vhdr)

    if raw_stand.ch_names != raw_walk.ch_names:
        raise RuntimeError(
            f"sub-{subject}: STAND and CS have different channel names"
        )
    if raw_stand.info["sfreq"] != raw_walk.info["sfreq"]:
        raise RuntimeError(
            f"sub-{subject}: STAND sfreq ({raw_stand.info['sfreq']}) != "
            f"CS sfreq ({raw_walk.info['sfreq']})"
        )

    stand_dur  = raw_stand.times[-1]
    walk_dur   = raw_walk.times[-1]
    walk_onset = float(raw_stand.n_times) / raw_stand.info["sfreq"]

    raw_concat = mne.concatenate_raws([raw_stand, raw_walk], preload=True)
    raw_concat.annotations.append(onset=0.0,        duration=stand_dur, description="STAND")
    raw_concat.annotations.append(onset=walk_onset, duration=walk_dur,  description="CS")

    print(f"  sub-{subject}: concat {raw_concat.n_times} samples  "
          f"sfreq={raw_concat.info['sfreq']:.0f} Hz")

    # ------------------------------------------------------------------ #
    # Stage 2 — Preprocess (parameterized by decisions)
    # ------------------------------------------------------------------ #
    if raw_concat.info["sfreq"] > TARGET_SFREQ:
        raw_concat.resample(TARGET_SFREQ, verbose=False)

    # High-pass filter (decision node: highpass_hz)
    highpass = decisions["highpass_hz"]
    raw_concat.filter(
        l_freq=float(highpass), h_freq=None,
        fir_design="firwin", verbose=False
    )

    # Low-pass filter (decision node: lowpass_hz — None means skip)
    lowpass = decisions["lowpass_hz"]
    if lowpass is not None:
        raw_concat.filter(
            l_freq=None, h_freq=float(lowpass),
            fir_design="firwin", verbose=False
        )

    # Notch
    raw_concat.notch_filter(freqs=LINE_FREQ, verbose=False)

    # Bad channels via PTP z-score
    data = raw_concat.get_data()
    ptp  = np.ptp(data, axis=1)
    z    = (ptp - ptp.mean()) / (ptp.std() + 1e-12)
    bads = [raw_concat.ch_names[i] for i in np.where(np.abs(z) > 3.0)[0]]
    print(f"  sub-{subject}: bad channels {bads}")
    raw_concat.info["bads"] = bads
    if bads:
        raw_concat.interpolate_bads(reset_bads=True)

    # ASR node (decision: use_asr); calibrate on STAND only
    stand_ann = [a for a in raw_concat.annotations
                 if a["description"] == "STAND"][0]
    calib = raw_concat.copy().crop(
        stand_ann["onset"],
        min(stand_ann["onset"] + stand_ann["duration"],
            raw_concat.times[-1])
    )
    calib = calib.crop(tmax=calib.times[-1] - 2.0)
    raw_concat = apply_asr_node(
        raw_concat, apply=decisions["use_asr"], calib_raw=calib
    )

    # Average reference
    raw_concat.set_eeg_reference("average", projection=False, verbose=False)

    # ------------------------------------------------------------------ #
    # Stage 3 — ICA (hybrid: load from branch_dir cache if available)
    # ------------------------------------------------------------------ #
    ica_path    = branch_dir / f"sub-{subject}_ica.fif"
    iclean_path = branch_dir / f"sub-{subject}_desc-icaClean_raw.fif"

    if iclean_path.exists() and ica_path.exists():
        print(f"  sub-{subject}: loading cached ICA from branch dir")
        raw_clean = mne.io.read_raw_fif(iclean_path, preload=True, verbose=False)
        ica       = mne.preprocessing.read_ica(ica_path)
        n_brain   = ica.n_components_ - len(ica.exclude)
    else:
        # ICA training epochs
        epochs_raw = mne.make_fixed_length_epochs(
            raw_concat, duration=EPOCH_DUR, preload=True, verbose=False
        )
        epochs_raw.pick("eeg")
        epochs_raw = drop_invalid_eeg_channels(epochs_raw)

        ar = AutoReject(
            n_interpolate=[1, 2, 4],
            random_state=RANDOM_STATE,
            verbose=False
        )
        ar.fit(epochs_raw)
        epochs_clean, _ = ar.transform(epochs_raw, return_log=True)

        if len(epochs_clean) < 20:
            raise RuntimeError(
                f"sub-{subject}: only {len(epochs_clean)} clean epochs "
                f"after AutoReject"
            )
        print(f"  sub-{subject}: {len(epochs_clean)} clean epochs")

        # Fit ICA (decision: n_components is fixed; brain_thresh controls labelling)
        ica = run_ica(
            epochs_clean,
            n_components=N_COMPONENTS,
            method="infomax",
            fit_params=dict(extended=True),
            random_state=RANDOM_STATE,
        )

        # ICLabel classification (decision: brain_thresh)
        result  = label_and_mark_ica(
            ica, epochs_clean,
            brain_thresh=float(decisions["brain_thresh"])
        )
        n_brain = len(result["brain_ics"])
        if n_brain == 0:
            raise RuntimeError(f"sub-{subject}: no brain ICs retained")

        print(f"  sub-{subject}: {n_brain} brain ICs  "
              f"{len(result['exclude_ics'])} excluded")

        ica.apply(raw_concat)

        # Drop channels with no valid sensor position after ICA
        pos = np.array([
            raw_concat.info["chs"][i]["loc"][:3]
            for i in range(len(raw_concat.ch_names))
        ])
        no_pos = [
            raw_concat.ch_names[i]
            for i in range(len(raw_concat.ch_names))
            if not np.any(pos[i] != 0)
        ]
        if no_pos:
            raw_concat.drop_channels(no_pos)
        if raw_concat.info["bads"]:
            raw_concat.interpolate_bads(reset_bads=True)

        raw_clean = raw_concat
        raw_clean.save(iclean_path, overwrite=True)
        ica.save(ica_path, overwrite=True)
        print(f"  sub-{subject}: ICA saved to branch dir")

    # ------------------------------------------------------------------ #
    # Stage 4 — Extract segments from cleaned raw
    # ------------------------------------------------------------------ #
    def crop_segment(raw, desc):
        ann   = [a for a in raw.annotations if a["description"] == desc][0]
        start = ann["onset"]
        stop  = min(ann["onset"] + ann["duration"], raw.times[-1])
        return raw.copy().crop(start, stop)

    raw_stand = crop_segment(raw_clean, "STAND")
    raw_walk  = crop_segment(raw_clean, "CS")
    raw_stand = raw_stand.crop(tmax=raw_stand.times[-1] - 2.0)
    sfreq     = raw_walk.info["sfreq"]
    ch_names  = raw_walk.ch_names

    # ------------------------------------------------------------------ #
    # Stage 5 — Standing baseline TFR (decision: baseline_type)
    # ------------------------------------------------------------------ #
    baseline_ok = True
    if decisions["baseline_type"] == "standing":
        stand_epochs = mne.make_fixed_length_epochs(
            raw_stand, duration=EPOCH_DUR, preload=True, verbose=False
        )
        stand_tfrs = []
        for ep in stand_epochs.get_data():
            if np.max(np.abs(ep)) > AMP_THRESH:
                continue
            power = mne.time_frequency.tfr_array_morlet(
                ep[np.newaxis], sfreq=sfreq, freqs=FREQS,
                n_cycles=N_CYCLES_WAV, output="power",
                zero_mean=True, verbose=False
            )[0]
            crop = int(EDGE_CROP * power.shape[-1])
            if crop > 0:
                power = power[..., crop:-crop]
            stand_tfrs.append(power)
        if len(stand_tfrs) == 0:
            raise RuntimeError(f"sub-{subject}: no clean standing epochs")
        stand_stack    = np.stack(stand_tfrs)
        baseline_power = stand_stack.mean(axis=(0, 3))   # (n_ch, n_freqs)
        baseline_mean  = float(baseline_power.mean())
        baseline_ok    = bool(1e-13 < baseline_mean < 1e-9)
        print(f"  sub-{subject}: {len(stand_tfrs)} standing epochs  "
              f"baseline_mean={baseline_mean:.3e}  ok={baseline_ok}")
    else:
        baseline_power = None   # computed from walking data after TFR

    # ------------------------------------------------------------------ #
    # Stage 6 — Gait cycle TFR
    # ------------------------------------------------------------------ #
    cycles    = pd.read_csv(event_dir / f"sub-{subject}_cycles.tsv", sep="\t")
    walk_data = raw_walk.get_data()

    tfr_cycles = []
    rto_fracs  = []

    for _, row in cycles.iterrows():
        i0 = int(round(row["rhs_start_s"] * sfreq))
        i1 = int(round(row["rhs_end_s"]   * sfreq))
        if i0 < 0 or i1 > walk_data.shape[1] or i1 <= i0:
            continue
        seg = walk_data[:, i0:i1]
        if not np.isfinite(seg).all() or np.max(np.abs(seg)) > AMP_THRESH:
            continue

        power = mne.time_frequency.tfr_array_morlet(
            seg[np.newaxis], sfreq=sfreq, freqs=FREQS,
            n_cycles=N_CYCLES_WAV, output="power",
            zero_mean=True, verbose=False
        )[0]
        crop = int(EDGE_CROP * power.shape[-1])
        if crop > 0:
            power = power[..., crop:-crop]
        x_old = np.linspace(0, 1, power.shape[-1])
        x_new = np.linspace(0, 1, N_POINTS)
        power = np.array([
            [np.interp(x_new, x_old, power[c, f])
             for f in range(power.shape[1])]
            for c in range(power.shape[0])
        ])
        tfr_cycles.append(power)
        dur = row["rhs_end_s"] - row["rhs_start_s"]
        rto_fracs.append(
            float(np.clip((row["rto_s"] - row["rhs_start_s"]) / dur, 0.3, 0.8))
        )

    if len(tfr_cycles) < 20:
        raise RuntimeError(
            f"sub-{subject}: only {len(tfr_cycles)} gait cycles accepted"
        )
    print(f"  sub-{subject}: {len(tfr_cycles)} gait cycles accepted")

    tfr_stack = np.stack(tfr_cycles)   # (n_cycles, n_ch, n_freqs, N_POINTS)

    # ------------------------------------------------------------------ #
    # Stage 7 — ERSP (decision: baseline_type)
    # ------------------------------------------------------------------ #
    if decisions["baseline_type"] == "standing":
        ersp_per_cycle = 10 * np.log10(
            tfr_stack / baseline_power[np.newaxis, :, :, np.newaxis]
        )
    else:
        # walking_mean: global mean power across all cycles and time points
        baseline_walk  = tfr_stack.mean(axis=(0, 3))   # (n_ch, n_freqs)
        baseline_mean  = float(baseline_walk.mean())
        baseline_ok    = bool(1e-13 < baseline_mean < 1e-9)
        ersp_per_cycle = 10 * np.log10(
            tfr_stack / baseline_walk[np.newaxis, :, :, np.newaxis]
        )

    # ------------------------------------------------------------------ #
    # Stage 8 — Phase split and t-statistic
    # ------------------------------------------------------------------ #
    if ROI_CHANNEL not in ch_names:
        raise RuntimeError(f"sub-{subject}: {ROI_CHANNEL} not found")
    cz = ch_names.index(ROI_CHANNEL)

    rto_indices = np.round(
        np.array(rto_fracs) * (N_POINTS - 1)
    ).astype(int)

    # Per-cycle mean beta at Cz, averaged across frequencies, for each phase
    stance_vals = np.array([
        ersp_per_cycle[k, cz, :, :rto_indices[k]].mean()
        for k in range(len(rto_indices))
    ])
    swing_vals = np.array([
        ersp_per_cycle[k, cz, :, rto_indices[k]:].mean()
        for k in range(len(rto_indices))
    ])

    # Two-tailed paired t-test across cycles: H0: stance beta == swing beta
    t_stat, t_pval = ttest_rel(stance_vals, swing_vals)

    print(f"  sub-{subject}: t={t_stat:.2f}  p={t_pval:.4f}  "
          f"stance={stance_vals.mean():+.2f}  swing={swing_vals.mean():+.2f}  "
          f"n_cycles={len(stance_vals)}")

    # ------------------------------------------------------------------ #
    # Stage 9 — Return
    # ------------------------------------------------------------------ #
    return {
        "subject":          subject,
        "t_stat":           float(t_stat),
        "t_pval":           float(t_pval),
        "beta_stance_mean": float(stance_vals.mean()),
        "beta_swing_mean":  float(swing_vals.mean()),
        "n_cycles":         len(stance_vals),
        "n_brain_ics":      int(n_brain),
        "baseline_ok":      bool(baseline_ok),
        **{k: v for k, v in decisions.items()},
    }
