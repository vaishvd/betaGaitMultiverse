import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks


# =========================================================
# FILTERING
# =========================================================
def lowpass(signal, fs, cutoff=6.0):

    b, a = butter(4, cutoff / (fs / 2), btype="low")

    return filtfilt(b, a, signal)


# =========================================================
# RELATIVE HEEL SIGNAL
# =========================================================
def heel_relative_signal(heel, pelvis):

    return heel - pelvis


# =========================================================
# EVENT DETECTION
# =========================================================
def detect_gait_events(signal, fs, cutoff=6.0):

    sig = lowpass(signal, fs, cutoff)

    vel = np.gradient(sig)

    min_dist = int(0.4 * fs)

    # Heel strike → velocity turns negative
    hs, _ = find_peaks(sig, distance=min_dist)

    # Toe-off → velocity turns positive
    to, _ = find_peaks(-sig, distance=min_dist)

    return hs, to, sig


# =========================================================
# QC REPORT
# =========================================================
def event_quality_report(lhs, lto, rhs, rto):

    return {
        "LHS": len(lhs),
        "LTO": len(lto),
        "RHS": len(rhs),
        "RTO": len(rto),
    }


# =========================================================
# RHS → RHS GAIT CYCLES
# =========================================================
def extract_rhs_cycles(
    rhs,
    fs,
    min_dur=0.5,
    max_dur=2.5,
):

    cycles = []

    for i in range(len(rhs) - 1):

        start = rhs[i]
        end   = rhs[i + 1]

        dur = (end - start) / fs

        if min_dur <= dur <= max_dur:

            cycles.append({
                "cycle_id": i,
                "rhs_start_sample": start,
                "rhs_end_sample": end,
                "rhs_start_s": start / fs,
                "rhs_end_s": end / fs,
                "duration_s": dur,
            })

    return pd.DataFrame(cycles)


# =========================================================
# SAVE EVENTS
# =========================================================
def build_events_dataframe(lhs, lto, rhs, rto, fs):

    rows = (
        [(s / fs, s, "LHS") for s in lhs] +
        [(s / fs, s, "LTO") for s in lto] +
        [(s / fs, s, "RHS") for s in rhs] +
        [(s / fs, s, "RTO") for s in rto]
    )

    return (
        pd.DataFrame(rows, columns=["onset_s", "sample", "event"])
        .sort_values("sample")
        .reset_index(drop=True)
    )


# =========================================================
# QC PLOT
# =========================================================
def plot_gait_qc(
    L_signal,
    R_signal,
    lhs,
    lto,
    rhs,
    rto,
    cycles_df,
    fs,
    out_file,
):

    t = np.arange(len(L_signal)) / fs

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
    )

    # -----------------------------------------------------
    # LEFT
    # -----------------------------------------------------
    ax[0].plot(
        t,
        L_signal,
        color="forestgreen",
        lw=1,
    )

    ax[0].scatter(
        lhs / fs,
        L_signal[lhs],
        color="magenta",
        label="LHS",
        s=30,
        zorder=5,
    )

    ax[0].scatter(
        lto / fs,
        L_signal[lto],
        color="blue",
        label="LTO",
        s=30,
        zorder=5,
    )

    ax[0].set_title("Left heel trajectory")
    ax[0].legend()


    # -----------------------------------------------------
    # RIGHT
    # -----------------------------------------------------
    ax[1].plot(
        t,
        R_signal,
        color="firebrick",
        lw=1,
    )

    ax[1].scatter(
        rhs / fs,
        R_signal[rhs],
        color="black",
        label="RHS",
        s=30,
        zorder=5,
    )

    ax[1].scatter(
        rto / fs,
        R_signal[rto],
        color="orange",
        label="RTO",
        s=30,
        zorder=5,
    )

    # Shade gait cycles
    for _, row in cycles_df.iterrows():

        ax[1].axvspan(
            row["rhs_start_s"],
            row["rhs_end_s"],
            color="limegreen",
            alpha=0.15,
        )

    ax[1].set_title("Right heel trajectory + RHS→RHS cycles")
    ax[1].legend()

    ax[1].set_xlabel("Time (s)")

    plt.tight_layout()

    plt.savefig(out_file, dpi=200)

    plt.close()