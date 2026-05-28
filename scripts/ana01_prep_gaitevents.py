import pandas as pd
from src.paths import get_dataset_dirs
from src.gait_cycles import (
    heel_relative_signal,
    detect_gait_events,
    build_events_dataframe,
    extract_valid_gait_cycles,
    event_quality_report,
    plot_gait_qc,
    plot_gait_segment,
)

DATASET = "stepup"
SUBJECT = "S1"

FS = 100

dirs = get_dataset_dirs(DATASET)

motion_file = (
    dirs["raw"]
    / f"sub-{SUBJECT}"
    / "motion"
    / f"sub-{SUBJECT}_task-CS.tsv"
)

events_out = (
    dirs["gait_events"]
    / f"sub-{SUBJECT}_events.tsv"
)

cycles_out = (
    dirs["gait_events"]
    / f"sub-{SUBJECT}_cycles.tsv"
)

qc_out = (
    dirs["gait_events"]
    / f"sub-{SUBJECT}_gait_qc.png"
)

# Load motion data

df = pd.read_csv(motion_file,sep="\t")

# Markers

LHEE = df["LHEE_PosY"].values
RHEE = df["RHEE_PosY"].values
PELV = df["SACR_PosY"].values


L_rel = heel_relative_signal(LHEE, PELV)

R_rel = heel_relative_signal(RHEE, PELV)

# Detect gait events

lhs, lto, L_filt = detect_gait_events(L_rel, FS)

rhs, rto, R_filt = detect_gait_events(R_rel, FS)

# Build events dataframe

events_df = build_events_dataframe(
    lhs,
    lto,
    rhs,
    rto,
    FS,
)

events_df.to_csv(
    events_out,
    sep="\t",
    index=False,
)

print(f"\nSaved events → {events_out.name}")

# Extract gait cycles

cycles_df = extract_valid_gait_cycles(
    lhs,
    lto,
    rhs,
    rto,
    FS,
)

cycles_df.to_csv(
    cycles_out,
    sep="\t",
    index=False,
)

print(f"Saved cycles → {cycles_out.name}")

# QC Summary

qc = event_quality_report(
    events_df,
    cycles_df,
)

print("\nQC summary")

for k, v in qc.items():
    print(f"{k}: {v}")

# QC Plot

plot_gait_qc(

    L_filt,
    R_filt,

    lhs,
    lto,
    rhs,
    rto,

    cycles_df,

    FS,

    qc_out,
)

print(f"\nSaved QC plot → {qc_out.name}")

# Plot a segment of the gait events

segment_out = dirs["gait_events"] / f"sub-{SUBJECT}_segment_10s.png"

plot_gait_segment(
    L_filt,
    R_filt,
    lhs,
    lto,
    rhs,
    rto,
    cycles_df,
    FS,
    segment_out,
    t_start=10.0,
)

print(f"Saved segment plot → {segment_out.name}")

print("\nDone.")