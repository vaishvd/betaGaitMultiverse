"""
ana03_ica2clean.py
==================
# ICA applied to the full concatenated (STAND + CS) preprocessed raw.
# Component rejection based on ICLabel (Pion-Tonachini et al. 2019
# NeuroImage) with brain probability threshold of 0.7.
# The cleaned concatenated raw is saved for downstream segment extraction.

Classify ICA components with ICLabel, reject artefact ICs,
and apply the cleaned decomposition to the concatenated continuous raw EEG.

Input
-----
d02_prep/
    sub-{sub}_concat_raw.fif         preprocessed concatenated (STAND+CS) raw
    sub-{sub}_ica.fif                fitted ICA (no exclusions)
    sub-{sub}_preica_clean_epo.fif   clean epochs (needed for ICLabel features)

Output
------
d03_clean/
    sub-{sub}_desc-icaClean_concat_raw.fif  ICA-cleaned concatenated raw
    sub-{sub}_clean-ica.fif                 ICA with exclusions
"""

import sys

import mne
import numpy as np

from src.paths import get_dataset_dirs
from src.config import DATASET, SUBJECTS
from src.pipeline_steps import apply_ica
from src.ica_utils import label_and_mark_ica
from src.qc import log_qc
from src.resume import stage_already_done

# Components below this threshold (or labelled as non-brain) are excluded.
BRAIN_THRESH = 0.7

# Optional: restrict this invocation to a single subject (see
# prepana02_raw2ica.py's identical mechanism -- one fresh process per
# subject, for datasets/machines where a long-lived process degrades
# across subjects).
if len(sys.argv) > 1:
    SUBJECTS = [sys.argv[1]]

dirs      = get_dataset_dirs(DATASET)
PREP_DIR  = dirs["prep"]
CLEAN_DIR = dirs["clean"]
QC_DIR    = dirs["qc"]

for subject in SUBJECTS:
    try:
        print(f"ICA CLEANING: sub-{subject}")

        ica_in_path = PREP_DIR / f"sub-{subject}_ica.fif"
        clean_ica_out = CLEAN_DIR / f"sub-{subject}_clean-ica.fif"
        clean_raw_out = CLEAN_DIR / f"sub-{subject}_desc-icaClean_concat_raw.fif"

        # Staleness is checked against ica_in_path (this stage's input):
        # if prepana02 refit the ICA more recently than this stage's own
        # output, that output is stale and must be redone.
        if stage_already_done(
            [clean_ica_out, clean_raw_out],
            inputs=[ica_in_path],
            validate=lambda: mne.io.read_raw_fif(clean_raw_out, preload=False, verbose=False),
        ):
            print(f"  Already complete -- skipping sub-{subject}")
            continue

        # Load preprocessed continuous raw

        raw = mne.io.read_raw_fif(
            PREP_DIR / f"sub-{subject}_concat_raw.fif",
            preload=True, verbose=False,
        )

        nan_frac = np.mean(np.isnan(raw.get_data()))

        # Load ICA

        ica = mne.preprocessing.read_ica(
            PREP_DIR / f"sub-{subject}_ica.fif",
            verbose=False,
        )
        print(f"\n  ICA components : {ica.n_components_}")

        # Load pre-ICA epochs

        epochs = mne.read_epochs(
            PREP_DIR / f"sub-{subject}_preica_clean_epo.fif",
            preload=True, verbose=False,
        )
        print(f"  Epochs for ICLabel: {len(epochs)} x {len(epochs.ch_names)} ch")

        # IC classification and rejection

        result = label_and_mark_ica(ica, epochs, brain_thresh=BRAIN_THRESH)

        print(f"\n  Brain ICs kept   : {len(result['brain_ics'])}  "
              f"{result['brain_ics']}")
        print(f"  Artefact ICs out : {len(result['exclude_ics'])}  "
              f"{result['exclude_ics']}")

        if len(result["brain_ics"]) == 0:
            print(f"  [FAIL] sub-{subject}: no brain ICs found -- skipping.")
            log_qc(
                qc_dir  = QC_DIR,
                subject = subject,
                stage   = "ica",
                flag    = "fail",
                metrics = {
                    "n_total_comps": len(ica.mixing_matrix_.T),
                    "n_brain_comps": 0,
                    "n_excl_comps":  len(ica.exclude),
                    "brain_frac":    0.0,
                    "note":          "no brain ICs found -- subject skipped",
                },
            )
            continue

        # Apply ICA, drop no-position channels, interpolate bads
        raw_clean = apply_ica(raw, ica, subject, iclean_path=None)

        # QC: verify no NaN introduced by ICA
        nan_out = np.mean(np.isnan(raw_clean.get_data()))
        if nan_out > 0:
            print("  ERROR: NaN in ICA output -- inspect component mixing matrix.")
            continue

        # Save ICA with exclusions for downstream scripts
        ica.save(clean_ica_out, overwrite=True, verbose=False)
        print(f"  Saved ICA with exclusions -> {clean_ica_out.name}")

        raw_clean.save(clean_raw_out, overwrite=True, verbose=False)
        print(f"  Saved -> {clean_raw_out.name}")

        # --- QC: ICA ---
        n_total_comps  = len(ica.mixing_matrix_.T)
        n_brain_comps  = len(result["brain_ics"])
        n_excl_comps   = len(ica.exclude)
        brain_frac     = n_brain_comps / n_total_comps if n_total_comps > 0 else 0.0

        if n_brain_comps == 0:
            ica_flag = "fail"
        elif n_brain_comps < 3 or brain_frac < 0.15:
            ica_flag = "warn"
        else:
            ica_flag = "pass"

        log_qc(
            qc_dir  = QC_DIR,
            subject = subject,
            stage   = "ica",
            flag    = ica_flag,
            metrics = {
                "n_total_comps": n_total_comps,
                "n_brain_comps": n_brain_comps,
                "n_excl_comps":  n_excl_comps,
                "brain_frac":    round(brain_frac, 3),
            },
        )
        print(f"  QC ica: {ica_flag}  "
              f"brain={n_brain_comps}/{n_total_comps}  "
              f"excluded={n_excl_comps}  "
              f"brain_frac={brain_frac:.3f}")

    except FileNotFoundError as e:
        print(f"\n  [SKIP] sub-{subject}: file not found -- {e}")
        continue
    except Exception as e:
        print(f"\n  [ERROR] sub-{subject}: unexpected error -- {e}")
        import traceback
        traceback.print_exc()
        continue

print("\nICA COMPLETE")
print(f"\nDone. Processed {len(SUBJECTS)} subject(s): {SUBJECTS}")
