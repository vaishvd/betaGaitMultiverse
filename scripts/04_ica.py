from pathlib import Path
from src.config import DIR_ICA, DIR_PREICA, DIR_SIGCLEAN
from src.ica_utils import load_epochs_ica, load_raw_for_ica, run_ica, apply_iclabel

SUBJECTS = ["S18"]
EPOCH_DATA = DIR_PREICA/ "sub-S18_preica_clean_epo.fif"
RAW_DATA = DIR_SIGCLEAN / f"sub-S18_clean_raw.fif"
OUTPUT_DIR = DIR_ICA

ICA_METHOD = "fastica"
N_COMPONENTS = 0.99  

for subject in SUBJECTS:
    print(f"\n{'='*60}\nProcessing subject {subject}\n{'='*60}")

    # Load epochs
    epochs = load_epochs_ica(subject, EPOCH_DATA)

    # Load raw data
    raw = load_raw_for_ica(subject, DIR_SIGCLEAN)

    # Ensure consistency 
    raw.pick(epochs.ch_names)

    # Fit ICA on epoched data
    ica = run_ica(epochs, method=ICA_METHOD, n_components=N_COMPONENTS)

    # Apply ICLabel
    epochs_clean, ica = apply_iclabel(ica, epochs)

    #Apply ICA to continuous data
    raw_clean = ica.apply(raw.copy())

    # Save outputs
    OUTPUT_DIR.mkdir(exist_ok=True)

    clean_fname = OUTPUT_DIR / f"sub-{subject}_desc-clean_epo.fif"
    epochs_clean.save(clean_fname, overwrite=True)
    raw_clean_fname = OUTPUT_DIR / f"sub-{subject}_desc-clean_raw.fif"

    ica_fname = OUTPUT_DIR / f"sub-{subject}_ica.fif"
    ica.save(ica_fname, overwrite=True)
    raw_clean.save(raw_clean_fname, overwrite=True)
    
    print(f"Saved cleaned epochs → {clean_fname.name}")
    print(f"Saved ICA solution → {ica_fname.name}")
    print(f"Saved cleaned raw data → {raw_clean_fname.name}")


print("\nICA pipeline complete.")