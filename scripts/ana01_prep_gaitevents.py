"""
ana01_prep_gaitevents.py
========================
Pipeline step 1: .tsv (motion) → Gait_events

Responsibilities:
  - Load motion capture data (with or without column headers)
  - Detect gait events (lhs, lto, rhs, rto)
  - Store events and cycle tables + QC plots
"""

import pandas as pd
from src.paths import get_dataset_dirs
from src.gait_cycles import (
    load_motion,
    heel_relative_signal,
    detect_gait_events,
    build_events_dataframe,
    extract_valid_gait_cycles,
    event_quality_report,
    plot_gait_qc,
    plot_gait_segment,
)

DATASET  = "stepup"
SUBJECTS = ["S1"]
FS       = 100

dirs = get_dataset_dirs(DATASET)

for SUBJECT in SUBJECTS:
    print(f"\n{'='*60}")
    print(f"  Subject: {SUBJECT}")
    print(f"{'='*60}")

    motion_file = (
        dirs["raw"]
        / f"sub-{SUBJECT}"
        / "motion"
        / f"sub-{SUBJECT}_task-CS.tsv"
    )
    events_out  = dirs["gait_events"] / f"sub-{SUBJECT}_events.tsv"
    cycles_out  = dirs["gait_events"] / f"sub-{SUBJECT}_cycles.tsv"
    qc_out      = dirs["gait_events"] / f"sub-{SUBJECT}_gait_qc.png"
    segment_out = dirs["gait_events"] / f"sub-{SUBJECT}_segment_10s.png"

    if not motion_file.exists():
        print(f"  Motion file not found, skipping: {motion_file}")
        continue

    # Load motion data — column names resolved from channels.tsv automatically
    df, marker_cols = load_motion(motion_file)
    print(f"  Loaded {len(df)} samples × {len(df.columns)} channels")

    # Extract marker signals using resolved column names
    LHEE = df[marker_cols["LHEE"]].values
    RHEE = df[marker_cols["RHEE"]].values
    PELV = df[marker_cols["PELV"]].values

    L_rel = heel_relative_signal(LHEE, PELV)
    R_rel = heel_relative_signal(RHEE, PELV)

    # Detect gait events
    lhs, lto, L_filt = detect_gait_events(L_rel, FS)
    rhs, rto, R_filt = detect_gait_events(R_rel, FS)

    # Build and save events dataframe
    events_df = build_events_dataframe(lhs, lto, rhs, rto, FS)
    events_df.to_csv(events_out, sep="\t", index=False)
    print(f"\n  Saved events  → {events_out.name}")

    # Extract and save gait cycles
    cycles_df = extract_valid_gait_cycles(lhs, lto, rhs, rto, FS)
    cycles_df.to_csv(cycles_out, sep="\t", index=False)
    print(f"  Saved cycles  → {cycles_out.name}")

    # QC summary
    qc = event_quality_report(events_df, cycles_df)
    print("\n  QC summary")
    for k, v in qc.items():
        print(f"    {k}: {v}")

    # QC plots
    plot_gait_qc(L_filt, R_filt, lhs, lto, rhs, rto, cycles_df, FS, qc_out)
    print(f"\n  Saved QC plot → {qc_out.name}")

    plot_gait_segment(
        L_filt, R_filt, lhs, lto, rhs, rto, cycles_df, FS, segment_out,
        t_start=10.0,
    )
    print(f"  Saved segment → {segment_out.name}")

print("\nDone.")