import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as mtransforms


# Gait event positions as % of the RHS-to-RHS cycle (population averages).
# RHS = 0% and 100%, RTO ~ 10%, LHS ~ 50%, LTO ~ 60%
# Stance phase: 0–60% (right foot on ground), Swing phase: 60–100%
STANCE_END_PCT = 60   # approximate end of right stance / start of right swing
RTO_PCT        = 10   # right toe-off
LHS_PCT        = 50   # left heel strike
LTO_PCT        = 60   # left toe-off  (= STANCE_END_PCT for right leg)


def plot_ersp_beta(
    ersp: np.ndarray,
    freqs: np.ndarray,
    ch_idx: int | list[int],
    ch_name: str | list[str],
    subject: str,
    out_file,
) -> None:
    """
    Publication-quality ERSP plot with gait phase annotations.

    Parameters
    ----------
    ersp     : (n_cycles, n_channels, n_freqs, n_timepoints)
    freqs    : frequency array, e.g. np.arange(13, 31)
    ch_idx   : single int or list of ints — channels to average over
    ch_name  : matching channel label(s) for the plot title
    subject  : subject ID string
    out_file : Path to save the figure
    """
    if isinstance(ch_idx, int):
        ch_idx = [ch_idx]
    if isinstance(ch_name, list):
        ch_name = "/".join(ch_name)

    # Average across cycles and channels → (n_freqs, n_timepoints)
    ersp_avg  = ersp[:, ch_idx, :, :].mean(axis=(0, 1))
    beta_mean = ersp_avg.mean(axis=0)
    x         = np.linspace(0, 100, ersp_avg.shape[-1])

    # Figure layout 
    fig = plt.figure(figsize=(11, 7))
    # Three rows: phase bar, heatmap, trace
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.06, 0.58, 0.36],
        width_ratios=[1, 0.03],
        hspace=0.08,
        wspace=0.04,
        left=0.09, right=0.88, top=0.91, bottom=0.10,
    )

    ax_bar   = fig.add_subplot(gs[0, 0])   # phase label bar
    ax_tf    = fig.add_subplot(gs[1, 0])   # time-frequency heatmap
    ax_cbar  = fig.add_subplot(gs[1, 1])   # colorbar
    ax_trace = fig.add_subplot(gs[2, 0])   # mean beta trace

    #Shared style 
    EVENT_COLOR  = "#444444"
    STANCE_COLOR = "#d6e8f7"   # light blue fill for stance
    SWING_COLOR  = "#fdebd0"   # light orange fill for swing
    FONT_LABEL   = dict(fontsize=8.5, color=EVENT_COLOR, ha="center", va="center")

    # Phase bar (top) 
    ax_bar.set_xlim(0, 100)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis("off")

    # Stance bar (0 → STANCE_END_PCT)
    ax_bar.axvspan(0, STANCE_END_PCT, ymin=0, ymax=1,
                   color=STANCE_COLOR, ec="#aac8e0", lw=0.8)
    ax_bar.text(STANCE_END_PCT / 2, 0.5, "Stance", **FONT_LABEL, fontweight="semibold")

    # Swing bar (STANCE_END_PCT → 100)
    ax_bar.axvspan(STANCE_END_PCT, 100, ymin=0, ymax=1,
                   color=SWING_COLOR, ec="#e0c8a0", lw=0.8)
    ax_bar.text((STANCE_END_PCT + 100) / 2, 0.5, "Swing", **FONT_LABEL, fontweight="semibold")

    # Heatmap 
    vlim = np.percentile(np.abs(ersp_avg), 95)
    im = ax_tf.imshow(
        ersp_avg,
        aspect="auto",
        origin="lower",
        extent=[0, 100, freqs[0], freqs[-1]],
        cmap="RdBu_r",
        vmin=-vlim,
        vmax= vlim,
    )
    ax_tf.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_tf.yaxis.set_major_locator(mticker.MultipleLocator(4))
    ax_tf.tick_params(labelbottom=False, bottom=False)
    ax_tf.set_xlim(0, 100)

    # Phase shading on heatmap
    ax_tf.axvspan(0,            STANCE_END_PCT, color=STANCE_COLOR, alpha=0.08)
    ax_tf.axvspan(STANCE_END_PCT, 100,          color=SWING_COLOR,  alpha=0.08)

    # Event lines on heatmap
    for pct, label in [(RTO_PCT, "RTO"), (LHS_PCT, "LHS"), (LTO_PCT, "LTO")]:
        ax_tf.axvline(pct, color=EVENT_COLOR, lw=0.9, ls="--", alpha=0.7)

    # RHS lines
    for pct in [0, 100]:
        ax_tf.axvline(pct, color=EVENT_COLOR, lw=1.3, ls="-")

    # Colorbar
    cb = fig.colorbar(im, cax=ax_cbar)
    cb.set_label("dB", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Mean beta trace 
    ax_trace.fill_between(x, beta_mean, 0,
                          where=(beta_mean >= 0), color="firebrick",  alpha=0.25)
    ax_trace.fill_between(x, beta_mean, 0,
                          where=(beta_mean <  0), color="steelblue",  alpha=0.30)
    ax_trace.plot(x, beta_mean, color="#1a1a2e", linewidth=1.6)
    ax_trace.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_trace.set_xlabel("Gait Cycle (%)", fontsize=10)
    ax_trace.set_ylabel("Mean dB", fontsize=10)
    ax_trace.set_xlim(0, 100)
    ax_trace.yaxis.set_major_locator(mticker.MaxNLocator(4, symmetric=True))

    # Phase shading on trace
    ylims = ax_trace.get_ylim()
    ax_trace.axvspan(0,            STANCE_END_PCT, color=STANCE_COLOR, alpha=0.25, zorder=0)
    ax_trace.axvspan(STANCE_END_PCT, 100,          color=SWING_COLOR,  alpha=0.25, zorder=0)

    # Event lines + labels on trace
    # Place labels just above the top of the axes using transData + offset
    for pct, label in [(RTO_PCT, "RTO"), (LHS_PCT, "LHS"), (LTO_PCT, "LTO")]:
        ax_trace.axvline(pct, color=EVENT_COLOR, lw=0.9, ls="--", alpha=0.7)
        ax_trace.text(pct, ylims[1] * 0.97, label,
                      fontsize=7.5, color=EVENT_COLOR,
                      ha="center", va="top",
                      bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.7))

    for pct, label in [(0, "RHS"), (100, "RHS")]:
        ax_trace.axvline(pct, color=EVENT_COLOR, lw=1.3, ls="-")
        ax_trace.text(pct + (1.5 if pct == 0 else -1.5), ylims[1] * 0.97, label,
                      fontsize=7.5, color=EVENT_COLOR,
                      ha="left" if pct == 0 else "right", va="top",
                      bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.7))

    # Title and save
    fig.suptitle(
        f"{subject}  —  Beta ERSP  ({ch_name})",
        fontsize=12, fontweight="semibold", y=0.97,
    )

    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()