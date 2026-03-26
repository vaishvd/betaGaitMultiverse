def run_pipeline(baseline):
    import mne
    from pathlib import Path
    from src.config import DIR_ICA
    from src.gait_cycles import extract_rhs_cycles, time_normalize
    from src.tfr import compute_tfr
    from src.ersp import baseline_per_cycle, baseline_global

    # Path to ICA-cleaned data
    sub = "S18"  
    raw_path = DIR_ICA / f"sub-{sub}_desc-clean_raw.fif"

    # Load the cleaned EEG
    raw = mne.io.read_raw_fif(raw_path, preload=True)

    # Now you’re working from the clean EEG
    # No ICA, no reprocessing
    
    cycles = extract_rhs_cycles(raw)
    cycles_norm = time_normalize(cycles)
    tfr = compute_tfr(cycles_norm)

    if baseline == "cycle":
        ersp = baseline_per_cycle(tfr)
    elif baseline == "global":
        ersp = baseline_global(tfr)
    else:
        raise ValueError("Unknown baseline type")

    beta_curve = ersp.mean(axis=0)  # average across cycles/channels
    beta_stance = beta_curve[:50].mean()
    beta_swing  = beta_curve[50:].mean()

    return {
        "beta_stance": float(beta_stance),
        "beta_swing": float(beta_swing),
        "beta_curve": beta_curve.tolist(),
    }