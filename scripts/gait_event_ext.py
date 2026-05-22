import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from src.config import DATASETS
from src.paths import get_dataset_dirs

DATASET = "stepup"
SUBJECT = "S1"

FS = 100
CUTOFF = 7
MIN_EVENT_DISTANCE = 0.5

# Get directories
dirs = get_dataset_dirs(DATASET)

raw_dir = dirs["raw"]
out_dir = dirs["gaitevents"]

motion_file = (
    raw_dir
    / f"sub-{SUBJECT}"
    / "motion"
    / f"sub-{SUBJECT}_task-CS.tsv"
)

events_out = (
    out_dir
    / f"sub-{SUBJECT}_task-CS_gaitevents.tsv"
)

qc_out = (
    out_dir
    / f"sub-{SUBJECT}_task-CS_qc.png"
)

# Functions

def lowpass(data, fs, cutoff=7):

    b, a = butter(
        4,
        cutoff / (fs / 2),
        btype="low"
    )

    return filtfilt(b, a, data)


def remove_close_events(events, min_samples):

    if len(events) == 0:
        return events

    cleaned = [events[0]]

    for e in events[1:]:

        if (e - cleaned[-1]) > min_samples:
            cleaned.append(e)

    return np.array(cleaned)


def detect_events(signal, fs):

    signal_filt = lowpass(signal, fs, CUTOFF)

    vel = np.gradient(signal_filt) * fs

    hs = np.where(
        (vel[:-1] > 0) &
        (vel[1:] <= 0)
    )[0]

    to = np.where(
        (vel[:-1] < 0) &
        (vel[1:] >= 0)
    )[0]

    min_samples = int(MIN_EVENT_DISTANCE * fs)

    hs = remove_close_events(hs, min_samples)
    to = remove_close_events(to, min_samples)

    return hs, to, signal_filt


print(f"\nLoading: {motion_file.name}")

df = pd.read_csv(
    motion_file,
    sep="\t"
)

print("\nColumns found:")
print(df.columns.tolist())

# Markers

LEFT_HEEL = "LHEE_PosX"
RIGHT_HEEL = "RHEE_PosX"
PELVIS = "SACR_PosX"

# Relative position of heels to pelvis (to account for forward progression during walking)

L_rel = df[LEFT_HEEL].values - df[PELVIS].values
R_rel = df[RIGHT_HEEL].values - df[PELVIS].values


# Detect gait events

lhs, lto, L_filt = detect_events(L_rel, FS)
rhs, rto, R_filt = detect_events(R_rel, FS)

print(f"\nLHS: {len(lhs)}")
print(f"LTO: {len(lto)}")
print(f"RHS: {len(rhs)}")
print(f"RTO: {len(rto)}")

# Combine events into a single DataFrame

events = []

for e in lhs:
    events.append([e / FS, e, "lhs"])

for e in lto:
    events.append([e / FS, e, "lto"])

for e in rhs:
    events.append([e / FS, e, "rhs"])

for e in rto:
    events.append([e / FS, e, "rto"])


events_df = pd.DataFrame(
    events,
    columns=["onset","sample","trial_type"]
)

events_df = events_df.sort_values("onset")
events_df.reset_index(drop=True, inplace=True)


# Save events

events_df.to_csv(
    events_out,
    sep="\t",
    index=False
)

print(f"\nSaved events:")
print(events_out)

# QC plot

t = np.arange(len(L_filt)) / FS

fig, ax = plt.subplots(
    2,
    1,
    figsize=(16, 8),
    sharex=True
)

# LEFT
ax[0].plot(t, L_filt)

ax[0].scatter(
    lhs / FS,
    L_filt[lhs],
    marker="v",
    label="LHS"
)

ax[0].scatter(
    lto / FS,
    L_filt[lto],
    marker="o",
    label="LTO"
)

ax[0].set_title("LEFT FOOT")
ax[0].legend()

# RIGHT
ax[1].plot(t, R_filt)

ax[1].scatter(
    rhs / FS,
    R_filt[rhs],
    marker="v",
    label="RHS"
)

ax[1].scatter(
    rto / FS,
    R_filt[rto],
    marker="o",
    label="RTO"
)

ax[1].set_title("RIGHT FOOT")
ax[1].legend()

ax[1].set_xlabel("Time (s)")

plt.tight_layout()

plt.savefig(
    qc_out,
    dpi=300
)

plt.close()

print(f"\nSaved QC:")
print(qc_out)

print("\nDone.")