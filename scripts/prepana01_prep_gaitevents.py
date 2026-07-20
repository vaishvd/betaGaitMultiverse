"""
ana01_prep_gaitevents.py
========================
Pipeline step 1: gait events -> cycles table.

Branches on config.EVENT_SOURCE:
  "mocap"      (stepUpAms)  : detect HS/TO from motion-capture heel/
                              pelvis markers (unchanged from prior
                              commits).
  "events_tsv" (Jacobsen)   : read pre-computed gait events from the
                              dataset's own BIDS events.tsv, restricted
                              to the non-button "easy" walking segment.

Both branches feed the same build_events_dataframe() /
extract_valid_gait_cycles() validation (LTO < LHS < RTO, exactly one of
each per RHS-RHS window) from src.gait_cycles, so prepana04 onward
never branches on dataset.
"""

import pandas as pd
from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS, EVENT_SOURCE
from src.qc import log_qc
from src.gait_cycles import (
    load_motion,
    heel_relative_signal,
    detect_gait_events,
    load_gait_events_from_tsv,
    build_events_dataframe,
    extract_valid_gait_cycles,
    event_quality_report,
    plot_gait_qc,
    plot_gait_segment,
)

FS = 100   # motion-capture sampling rate (stepUpAms "mocap" branch only)

if EVENT_SOURCE == "events_tsv":
    from src.config import (
        TASK_NAME, EEG_SFREQ_NATIVE, SEGMENT_START_VALUE,
        SEGMENT_END_VALUE, GAIT_EVENT_VALUES,
    )

dirs = get_dataset_dirs(DATASET)
QC_DIR = dirs["qc"]

for SUBJECT in SUBJECTS:
    try:
        print(f"\n{'='*60}")
        print(f"  Subject: {SUBJECT}")
        print(f"{'='*60}")

        events_out  = dirs["gait_events"] / f"sub-{SUBJECT}_events.tsv"
        cycles_out  = dirs["gait_events"] / f"sub-{SUBJECT}_cycles.tsv"
        qc_out      = dirs["gait_events"] / f"sub-{SUBJECT}_gait_qc.png"
        segment_out = dirs["gait_events"] / f"sub-{SUBJECT}_segment_10s.png"

        do_plots = False
        L_filt = R_filt = lhs = lto = rhs = rto = None

        if EVENT_SOURCE == "mocap":
            motion_file = (
                dirs["raw"]
                / f"sub-{SUBJECT}"
                / "motion"
                / f"sub-{SUBJECT}_task-CS.tsv"
            )
            if not motion_file.exists():
                print(f"  Motion file not found, skipping: {motion_file}")
                continue

            # Load motion data -- column names resolved from channels.tsv automatically
            df, marker_cols = load_motion(motion_file)
            print(f"  Loaded {len(df)} samples x {len(df.columns)} channels")

            # Extract marker signals using resolved column names
            LHEE = df[marker_cols["LHEE"]].values
            RHEE = df[marker_cols["RHEE"]].values
            PELV = df[marker_cols["PELV"]].values

            L_rel = heel_relative_signal(LHEE, PELV)
            R_rel = heel_relative_signal(RHEE, PELV)

            # Detect gait events
            lhs, lto, L_filt = detect_gait_events(L_rel, FS)
            rhs, rto, R_filt = detect_gait_events(R_rel, FS)
            fs_for_cycles = FS
            do_plots = True

        elif EVENT_SOURCE == "events_tsv":
            events_tsv = (
                dirs["raw"] / f"sub-{SUBJECT}" / "eeg"
                / f"sub-{SUBJECT}_task-{TASK_NAME}_events.tsv"
            )
            if not events_tsv.exists():
                print(f"  events.tsv not found, skipping: {events_tsv}")
                continue

            lhs, lto, rhs, rto, seg_t0, seg_t1 = load_gait_events_from_tsv(
                events_tsv, SEGMENT_START_VALUE, SEGMENT_END_VALUE,
                GAIT_EVENT_VALUES,
            )
            fs_for_cycles = EEG_SFREQ_NATIVE
            print(f"  Segment [{seg_t0:.1f}, {seg_t1:.1f}] s  "
                  f"events: LHS={len(lhs)} LTO={len(lto)} "
                  f"RHS={len(rhs)} RTO={len(rto)}")

        else:
            raise ValueError(f"Unknown EVENT_SOURCE: {EVENT_SOURCE!r}")

        # Build and save events dataframe
        events_df = build_events_dataframe(lhs, lto, rhs, rto, fs_for_cycles)
        events_df.to_csv(events_out, sep="\t", index=False)
        print(f"\n  Saved events  -> {events_out.name}")

        # Extract and save gait cycles
        cycles_df = extract_valid_gait_cycles(lhs, lto, rhs, rto, fs_for_cycles)
        cycles_df.to_csv(cycles_out, sep="\t", index=False)
        print(f"  Saved cycles  -> {cycles_out.name}")

        #  QC: gait events
        n_cycles = len(cycles_df)
        mean_dur = float(cycles_df["duration_s"].mean()) if n_cycles > 0 else 0.0
        std_dur  = float(cycles_df["duration_s"].std())  if n_cycles > 0 else 0.0

        # Cycle duration coefficient of variation (CV) is a standard gait
        # variability metric. CV > 0.10 indicates high stride-to-stride
        # variability. LTO fraction is not used for QC -- its value (~0.10)
        # reflects normal right-foot cycle timing and is not a quality indicator.
        # See: Hausdorff et al. 2001 J Appl Physiol
        cv_dur = std_dur / mean_dur if mean_dur > 0 else 1.0

        if n_cycles < 30:
            gait_flag = "fail"
        elif n_cycles < 60 or cv_dur > 0.15:
            gait_flag = "warn"
        else:
            gait_flag = "pass"

        log_qc(
            qc_dir    = QC_DIR,
            subject   = SUBJECT,
            stage     = "gait_events",
            flag      = gait_flag,
            metrics   = {
                "n_cycles":   n_cycles,
                "mean_dur_s": round(mean_dur, 3),
                "std_dur_s":  round(std_dur, 3),
                "cv_dur":     round(cv_dur, 3),
            },
        )
        print(f"  QC gait_events: {gait_flag}  "
              f"n={n_cycles}  dur={mean_dur:.3f}+/-{std_dur:.3f}s  "
              f"cv={cv_dur:.3f}")

        # QC summary
        qc = event_quality_report(events_df, cycles_df)
        print("\n  QC summary")
        for k, v in qc.items():
            print(f"    {k}: {v}")

        # QC plots -- need a continuous kinematic trace to plot against,
        # only available in the "mocap" branch (Jacobsen provides no
        # heel/pelvis position signal, only discrete event timestamps).
        if do_plots:
            plot_gait_qc(L_filt, R_filt, lhs, lto, rhs, rto, cycles_df, FS, qc_out)
            print(f"\n  Saved QC plot -> {qc_out.name}")

            plot_gait_segment(
                L_filt, R_filt, lhs, lto, rhs, rto, cycles_df, FS, segment_out,
                t_start=10.0,
            )
            print(f"  Saved segment -> {segment_out.name}")
        else:
            print("\n  QC plots skipped (no continuous kinematic trace "
                  "for this dataset's event source).")

    except FileNotFoundError as e:
        print(f"\n  [SKIP] sub-{SUBJECT}: file not found -- {e}")
        continue
    except Exception as e:
        print(f"\n  [ERROR] sub-{SUBJECT}: unexpected error -- {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\nDone. Processed {len(SUBJECTS)} subject(s): {SUBJECTS}")
