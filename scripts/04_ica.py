from pathlib import Path
from src.config import DIR_ICA, DIR_DATA
from src.ica_utils import load_data_ica, run_ica, apply_iclabel

SUBJECTS = ["S18"]
DATA_DIR = DIR_DATA/ "d03_preica"
OUTPUT_DIR = DIR_ICA

ICA_METHOD = "fastica"
N_COMPONENTS = 0  # 0 = all components
L_FREQ = 1.0
BAD_CHAN_THRESHOLD = 3.0

for subject in SUBJECTS:
    print(f"\n{'='*60}\nProcessing subject {subject}\n{'='*60}")

    # Load raw EEG and reproduce pre-ICA preprocessing
    raw = load_data_ica(subject, DATA_DIR, l_freq=L_FREQ, bad_chan_threshold=BAD_CHAN_THRESHOLD)

    # Fit ICA
    ica = run_ica(raw, method=ICA_METHOD, n_components=N_COMPONENTS)

    # Apply ICLabel to classify and exclude non-brain components
    raw_clean, ica = apply_iclabel(ica, raw)

    # Save cleaned data and ICA solution
    OUTPUT_DIR.mkdir(exist_ok=True)
    clean_fname = OUTPUT_DIR / f"sub-{subject}_preica_clean_raw.fif"
    raw_clean.save(clean_fname, overwrite=True)
    ica_fname = OUTPUT_DIR / f"sub-{subject}_ica.fif"
    ica.save(ica_fname)
    
    print(f"Saved cleaned raw → {clean_fname.name}")
    print(f"Saved ICA solution → {ica_fname.name}")

print("\nICA pipeline complete.")