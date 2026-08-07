"""
ana02_raw2ica.py
================
# Standing (STAND) and walking (CS) recordings are concatenated before
# preprocessing so that filtering, referencing, bad-channel interpolation,
# and ICA decomposition are applied identically to both conditions.
# ICA is fitted on epochs drawn from the full concatenated recording,
# giving more data for decomposition and ensuring the same component
# structure is used for both conditions.
# Condition boundaries are stored as annotations ("STAND", "CS") in the
# concatenated raw for downstream segment extraction.
# See: Makeig et al. 1996 J Neurosci; Delorme et al. 2012 Front Hum Neurosci

Input
-----
d00_raw/  dataset-specific raw recording(s), loaded and annotated with
          "STAND"/"CS" regions by src.pipeline_steps.load_and_concatenate
          (BrainVision sub-{sub}_task-STAND.vhdr + _task-CS.vhdr for
          stepup; one continuous EEGLAB .set for Jacobsen ds003039)

Output
------
d02_prep/
    sub-{sub}_concat_raw.fif           preprocessed concatenated (STAND+CS) raw
    sub-{sub}_preica_clean_epo.fif     AutoReject-cleaned epochs for ICA fit
    sub-{sub}_ica.fif                  fitted ICA (no components excluded)
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS, USE_ASR, ASR_CUTOFF
from src.pipeline_steps import load_and_concatenate, preprocess_raw, fit_ica
from src.ica_utils import save_ica_component_plots
from src.qc import log_qc
from src.resume import stage_already_done

# Optional: restrict this invocation to a single subject, e.g.
#   python scripts/prepana02_raw2ica.py 008
# Lets a caller run one subject per fresh interpreter (see the ASR=20
# Jacobsen re-run, where a single long-lived process degraded across
# subjects -- escalating AutoReject+ICA time per subject, 43->125min over
# 7 subjects, then 700+min with no output on the 8th). Default (no arg)
# is unchanged: process every subject in SUBJECTS, as always.
if len(sys.argv) > 1:
    SUBJECTS = [sys.argv[1]]

L_FREQ       = 1.0   # Hz — canonical pipeline high-pass
H_FREQ       = 60.0  # Hz — canonical pipeline low-pass
BRAIN_THRESH = 0.7   # ICLabel brain probability threshold

dirs     = get_dataset_dirs(DATASET)
RAW_DIR  = dirs["raw"]
PREP_DIR = dirs["prep"]
QC_DIR   = dirs["qc"]

for subject in SUBJECTS:
    try:
        print(f"\nProcessing sub-{subject}")

        concat_out    = PREP_DIR / f"sub-{subject}_concat_raw.fif"
        ica_save_path = PREP_DIR / f"sub-{subject}_ica.fif"
        epo_save_path = PREP_DIR / f"sub-{subject}_preica_clean_epo.fif"

        # ica_save_path/epo_save_path must also be NEWER than concat_out:
        # a fresh concat_raw.fif sitting next to a stale ica.fif (e.g.
        # left over from an interrupted prior run) must NOT be treated
        # as complete -- see src.resume.stage_already_done.
        if stage_already_done(
            [ica_save_path, epo_save_path],
            inputs=[concat_out],
            validate=lambda: mne.preprocessing.read_ica(ica_save_path, verbose=False),
        ):
            print(f"  Already complete -- skipping sub-{subject}")
            continue

        # load_and_concatenate() raises FileNotFoundError for whichever
        # dataset-specific raw file is missing (BrainVision .vhdr or
        # EEGLAB .set, see src.pipeline_steps), caught below.
        raw_concat = load_and_concatenate(subject, RAW_DIR)

        raw_concat = preprocess_raw(
            raw_concat,
            subject,
            highpass_hz = 1.0,
            lowpass_hz  = H_FREQ,
            use_asr     = USE_ASR,
            asr_cutoff  = ASR_CUTOFF,
            use_gedai   = False,   # GEDAI not active in main pipeline
        )

        # Save preprocessed concatenated raw

        raw_concat.save(concat_out, overwrite=True)
        print(f"  Saved concat raw -> {concat_out.name}")

        # Save electrode montage for reference (standard_1005 layout)
        try:
            import matplotlib.pyplot as plt
            fig = raw_concat.plot_sensors(
                show_names=True,
                title=f"sub-{subject} — standard_1005 montage",
                show=False,
            )
            montage_path = PREP_DIR / f"sub-{subject}_montage.png"
            fig.savefig(montage_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Montage saved -> {montage_path.name}")
        except Exception as e:
            print(f"  [WARN] Montage save failed: {e}")

        # ICA: AutoReject → Extended Infomax → ICLabel → apply
        # Saves ICA object and pre-ICA epochs to disk for prepana03
        raw_concat, ica, n_brain_ics = fit_ica(
            raw_concat,
            subject,
            brain_thresh = BRAIN_THRESH,
            ica_path     = ica_save_path,
            iclean_path  = None,
            epo_path     = epo_save_path,
        )

        save_ica_component_plots(ica, PREP_DIR, subject)

        # --- QC: preprocessing ---
        concat_dur_s = raw_concat.times[-1]
        n_ica_comps  = ica.n_components_

        if n_brain_ics == 0 or n_ica_comps < 10:
            prep_flag = "fail"
        elif n_brain_ics < 3:
            prep_flag = "warn"
        else:
            prep_flag = "pass"

        log_qc(
            qc_dir  = QC_DIR,
            subject = subject,
            stage   = "preprocessing",
            flag    = prep_flag,
            metrics = {
                "concat_dur_s": round(float(concat_dur_s), 1),
                "n_ica_comps":  n_ica_comps,
                "n_brain_ics":  n_brain_ics,
                "n_excl_ics":   len(ica.exclude),
                "use_asr":      bool(USE_ASR),
                "asr_cutoff":   float(ASR_CUTOFF) if USE_ASR else None,
            },
        )
        print(f"  QC preprocessing: {prep_flag}  "
              f"dur={concat_dur_s:.0f}s  "
              f"ica_comps={n_ica_comps}  brain_ics={n_brain_ics}  "
              f"excl={len(ica.exclude)}")

    except FileNotFoundError as e:
        print(f"\n  [SKIP] sub-{subject}: file not found -- {e}")
        continue
    except Exception as e:
        print(f"\n  [ERROR] sub-{subject}: unexpected error -- {e}")
        import traceback
        traceback.print_exc()
        continue

print("\nICA FIT COMPLETE")
print(f"\nDone. Processed {len(SUBJECTS)} subject(s): {SUBJECTS}")
