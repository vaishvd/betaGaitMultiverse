import pandas as pd

from src.paths import get_dataset_dirs

from src.gait_cycles import (
    heel_relative_signal,
    detect_gait_events,
    extract_rhs_cycles,
    build_events_dataframe,
    event_quality_report,
    plot_gait_qc,
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

df = pd.read_csv(motion_file, sep="\t")

# Markers

LHEE = df["LHEE_PosX"].values
RHEE = df["RHEE_PosX"].values
PELV = df["SACR_PosX"].values


L_rel = heel_relative_signal(LHEE, PELV)
R_rel = heel_relative_signal(RHEE, PELV)

# Event detection
lhs, lto, L_filt = detect_gait_events(L_rel, FS)

rhs, rto, R_filt = detect_gait_events(R_rel, FS)

#QC

qc = event_quality_report(
    lhs,
    lto,
    rhs,
    rto,
)

print("\nDetected gait events:")
print(qc)

#Events dataframe

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
cycles_df = extract_rhs_cycles(
    rhs,
    FS,
)

cycles_df.to_csv(
    cycles_out,
    sep="\t",
    index=False,
)

print(f"Saved cycles → {cycles_out.name}")


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

print(f"Saved QC plot → {qc_out.name}")

print("\nDone.")