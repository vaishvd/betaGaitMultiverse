import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks


# Filtering

def lowpass(signal, fs, cutoff=6.0):

    b, a = butter(
        4,
        cutoff / (fs / 2),
        btype="low",
    )

    return filtfilt(b, a, signal)


#Relative heel signal

def heel_relative_signal(heel, pelvis):

    sig = heel - pelvis
    sig = sig - np.mean(sig)

    return sig


# Gait event detection

def detect_gait_events(
    signal,
    fs,
    cutoff=6.0,
    min_step_time=0.5,
):
    sig = lowpass(signal, fs, cutoff)

    min_dist   = int(min_step_time * fs)
    prominence = np.std(sig) * 1.0   # conservative: avoids small spurious peaks

    # Heel strike = positive peak
    hs, _ = find_peaks(sig, distance=min_dist, prominence=prominence)

    # Toe off = negative peak (trough)
    to, _ = find_peaks(-sig, distance=min_dist, prominence=prominence)

    return hs, to, sig


# Build events dataframe

def build_events_dataframe(lhs, lto, rhs, rto, fs):

    rows = (
        [(s / fs, s, "LHS") for s in lhs] +
        [(s / fs, s, "LTO") for s in lto] +
        [(s / fs, s, "RHS") for s in rhs] +
        [(s / fs, s, "RTO") for s in rto]
    )

    df = (
        pd.DataFrame(rows, columns=["onset_s", "sample", "event"])
        .sort_values("sample")
        .reset_index(drop=True)
    )

    return df


# =========================================================
# LEVEL-WALKING CYCLE EXTRACTION
# RHS → LTO → LHS → RTO → RHS  (standard walking order)
#
# Strategy
# --------
# 1. Remove spurious RHS peaks: any RHS < min_stride apart from its
#    predecessor AND with no LHS in between is a double-detection and
#    is dropped before cycle search.
# 2. For each remaining RHS-RHS window check:
#    a. Exactly one LTO, one LHS, one RTO inside the window.
#    b. Strict walking order: LTO < LHS < RTO.
#    c. LTO falls within the first lto_max_frac of the cycle
#       (walking LTO ≈ 10-15 %; step-up LTO ≈ 50 %).
#    d. Duration within [min_dur, max_dur].
# =========================================================

def _remove_false_rhs(rhs, lhs, fs, min_stride=0.8):
    """Drop RHS events whose gap from the previous kept RHS is < min_stride
    AND which contain no LHS event in that gap (double-peak artefact)."""

    rhs  = np.sort(rhs)
    lhs  = np.sort(lhs)
    keep = np.ones(len(rhs), dtype=bool)

    prev = 0   # index of last kept event
    for i in range(1, len(rhs)):
        if not keep[prev]:
            prev = i
            continue
        gap = (rhs[i] - rhs[prev]) / fs
        if gap < min_stride:
            lhs_between = lhs[(lhs > rhs[prev]) & (lhs < rhs[i])]
            if len(lhs_between) == 0:
                keep[i] = False     # remove the short-interval RHS
                continue            # prev stays the same
        prev = i

    return rhs[keep]


def extract_valid_gait_cycles(
    lhs,
    lto,
    rhs,
    rto,
    fs,
    min_dur=0.8,
    max_dur=2.0,
    lto_max_frac=0.30,
):
    lhs = np.sort(lhs)
    lto = np.sort(lto)
    rhs = np.sort(rhs)
    rto = np.sort(rto)

    print(f"\nDetected events:  RHS={len(rhs)}  LTO={len(lto)}  "
          f"LHS={len(lhs)}  RTO={len(rto)}")

    # ── step 1: remove double-detected RHS peaks ────────────────────
    rhs_clean = _remove_false_rhs(rhs, lhs, fs)
    n_removed = len(rhs) - len(rhs_clean)
    print(f"  RHS after artefact removal: {len(rhs_clean)}  "
          f"({n_removed} spurious peaks dropped)")

    if len(rhs_clean) < 2:
        print("Fewer than 2 RHS events after cleaning — cannot extract cycles.")
        return pd.DataFrame()

    cycles            = []
    rej_count         = 0   # wrong number of events in window
    rej_order         = 0   # LTO/LHS/RTO not in walking order
    rej_early_lto     = 0   # LTO too late (step-up, not walking)
    rej_duration      = 0   # cycle duration out of range

    for i in range(len(rhs_clean) - 1):

        rhs1 = rhs_clean[i]
        rhs2 = rhs_clean[i + 1]

        # ── a. collect events in window ─────────────────────────────
        lto_i = lto[(lto > rhs1) & (lto < rhs2)]
        lhs_i = lhs[(lhs > rhs1) & (lhs < rhs2)]
        rto_i = rto[(rto > rhs1) & (rto < rhs2)]

        if len(lto_i) != 1 or len(lhs_i) != 1 or len(rto_i) != 1:
            rej_count += 1
            continue

        lto_ev = lto_i[0]
        lhs_ev = lhs_i[0]
        rto_ev = rto_i[0]

        # ── b. strict walking order: LTO < LHS < RTO ────────────────
        if not (lto_ev < lhs_ev < rto_ev):
            rej_order += 1
            continue

        dur = (rhs2 - rhs1) / fs

        # ── c. LTO timing: must be early in cycle (walking) ─────────
        lto_frac = (lto_ev - rhs1) / (rhs2 - rhs1)
        if lto_frac > lto_max_frac:
            rej_early_lto += 1
            continue

        # ── d. duration ──────────────────────────────────────────────
        if not (min_dur <= dur <= max_dur):
            rej_duration += 1
            continue

        cycles.append({
            "cycle_id":    len(cycles),
            "rhs_start":   rhs1,
            "lto":         lto_ev,
            "lhs":         lhs_ev,
            "rto":         rto_ev,
            "rhs_end":     rhs2,
            "rhs_start_s": rhs1 / fs,
            "lto_s":       lto_ev / fs,
            "lhs_s":       lhs_ev / fs,
            "rto_s":       rto_ev / fs,
            "rhs_end_s":   rhs2 / fs,
            "duration_s":  dur,
            "lto_frac":    round(lto_frac, 3),
        })

    cycles_df = pd.DataFrame(cycles)

    print(f"\nCycle QC  (level-walking: RHS → LTO → LHS → RTO → RHS)")
    print(f"  Windows checked          : {len(rhs_clean) - 1}")
    print(f"  Valid cycles             : {len(cycles_df)}")
    print(f"  Rejected — event count   : {rej_count}")
    print(f"  Rejected — wrong order   : {rej_order}")
    print(f"  Rejected — late LTO      : {rej_early_lto}  "
          f"(LTO > {lto_max_frac*100:.0f}% of cycle = step-up kinematics)")
    print(f"  Rejected — duration      : {rej_duration}")

    if len(cycles_df) > 0:
        print(f"\n  LTO timing in valid cycles (% of cycle):")
        print(f"    mean {cycles_df['lto_frac'].mean()*100:.1f}%  "
              f"range [{cycles_df['lto_frac'].min()*100:.1f}%, "
              f"{cycles_df['lto_frac'].max()*100:.1f}%]")

    return cycles_df


# QC Summary

def event_quality_report(events_df, cycles_df):

    return {
        "total_events": len(events_df),
        "RHS":          int((events_df["event"] == "RHS").sum()),
        "LTO":          int((events_df["event"] == "LTO").sum()),
        "LHS":          int((events_df["event"] == "LHS").sum()),
        "RTO":          int((events_df["event"] == "RTO").sum()),
        "valid_cycles": len(cycles_df),
    }


# QC Plot

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

    fig = plt.figure(figsize=(16, 10))
    ax0 = fig.add_subplot(3, 1, 1)
    ax1 = fig.add_subplot(3, 1, 2, sharex=ax0)
    ax2 = fig.add_subplot(3, 1, 3)   # histogram — independent x-axis

    # ── left heel ────────────────────────────────────────────────────
    ax0.plot(t, L_signal, color="forestgreen", lw=0.8)
    ax0.scatter(lhs / fs, L_signal[lhs], color="magenta",
                s=25, zorder=3, label="LHS")
    ax0.scatter(lto / fs, L_signal[lto], color="royalblue",
                s=25, zorder=3, label="LTO")
    ax0.set_ylabel("Rel. position (m)")
    ax0.set_title("Left heel AP trajectory (pelvis-referenced)")
    ax0.legend(loc="upper right", fontsize=8)

    # ── right heel ───────────────────────────────────────────────────
    ax1.plot(t, R_signal, color="firebrick", lw=0.8)
    ax1.scatter(rhs / fs, R_signal[rhs], color="black",
                s=25, zorder=3, label="RHS")
    ax1.scatter(rto / fs, R_signal[rto], color="darkorange",
                s=25, zorder=3, label="RTO")

    for _, row in cycles_df.iterrows():
        ax1.axvspan(row["rhs_start_s"], row["rhs_end_s"],
                    color="limegreen", alpha=0.20)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Rel. position (m)")
    ax1.set_title(
        f"Right heel AP trajectory  |  valid level-walking cycles "
        f"(n={len(cycles_df)}, green)"
    )
    ax1.legend(loc="upper right", fontsize=8)

    # ── duration histogram ───────────────────────────────────────────
    if len(cycles_df) > 0:
        ax2.hist(cycles_df["duration_s"], bins=20,
                 color="steelblue", edgecolor="white")
        mean_dur = cycles_df["duration_s"].mean()
        ax2.axvline(mean_dur, color="red", linestyle="--",
                    label=f"mean {mean_dur:.2f} s")
        ax2.legend(fontsize=8)
        print(f"\nCycle duration summary:")
        print(cycles_df["duration_s"].describe().to_string())
    else:
        ax2.text(0.5, 0.5,
                 "No level-walking cycles detected\n"
                 "(all cycles have step-up kinematics: LTO > 30% of cycle)",
                 ha="center", va="center", fontsize=11,
                 transform=ax2.transAxes, color="firebrick")

    ax2.set_xlabel("Gait cycle duration (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of valid gait cycle durations")

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


# Plot a segment of the gait events

def plot_gait_segment(
    L_signal,
    R_signal,
    lhs,
    lto,
    rhs,
    rto,
    cycles_df,
    fs,
    out_file,
    t_start=10.0,
    duration=10.0,
):
    t = np.arange(len(L_signal)) / fs

    i0 = int(t_start * fs)
    i1 = int((t_start + duration) * fs)
    t_seg = t[i0:i1]

    def in_seg(idx):
        return idx[(idx >= i0) & (idx < i1)]

    lhs_s = in_seg(lhs)
    lto_s = in_seg(lto)
    rhs_s = in_seg(rhs)
    rto_s = in_seg(rto)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax0.plot(t_seg, L_signal[i0:i1], color="forestgreen", lw=1.2)
    ax0.scatter(lhs_s / fs, L_signal[lhs_s], color="magenta",
                s=60, zorder=3, label="LHS")
    ax0.scatter(lto_s / fs, L_signal[lto_s], color="royalblue",
                s=60, zorder=3, label="LTO")
    ax0.set_ylabel("Rel. position (m)")
    ax0.set_title(f"Left heel AP  |  t = {t_start}–{t_start+duration} s")
    ax0.legend(loc="upper right", fontsize=9)

    ax1.plot(t_seg, R_signal[i0:i1], color="firebrick", lw=1.2)
    ax1.scatter(rhs_s / fs, R_signal[rhs_s], color="black",
                s=60, zorder=3, label="RHS")
    ax1.scatter(rto_s / fs, R_signal[rto_s], color="darkorange",
                s=60, zorder=3, label="RTO")

    if len(cycles_df) > 0:
        mask = (
            (cycles_df["rhs_start_s"] >= t_start) &
            (cycles_df["rhs_end_s"]   <= t_start + duration)
        )
        for _, row in cycles_df[mask].iterrows():
            ax1.axvspan(row["rhs_start_s"], row["rhs_end_s"],
                        color="limegreen", alpha=0.20)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Rel. position (m)")
    ax1.set_title("Right heel AP  |  valid cycles shaded")
    ax1.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()