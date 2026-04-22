import mne
from mne_icalabel import label_components
from src.config import DIR_PREICA, DIR_SIGCLEAN, DIR_ICA

INPUT_EPOCHS = DIR_PREICA
INPUT_RAW    = DIR_SIGCLEAN
OUTPUT_DIR   = DIR_ICA

ICA_METHOD    = "fastica"
N_COMPONENTS  = 0.99

# Keep only "brain" components above this confidence
BRAIN_LABEL = "brain"
BRAIN_THRESHOLD = 0.7   
SUBJECTS = ["S18"]

for subject in SUBJECTS:
    print(f"\n{'='*50}\nProcessing {subject}\n{'='*50}")

    # Load data
    epochs = mne.read_epochs(
        INPUT_EPOCHS / f"sub-{subject}_preica_clean_epo.fif", preload=True
    ).pick("eeg")

    raw = mne.io.read_raw_fif(
        INPUT_RAW / f"sub-{subject}_clean_raw.fif", preload=True
    ).pick(epochs.ch_names)

    # Fit ICA 
    rank = mne.compute_rank(epochs, rank="info")
    print(f"  Data rank: {rank}")

    ica = mne.preprocessing.ICA(
        n_components=N_COMPONENTS,
        method=ICA_METHOD,
        random_state=42,
        max_iter="auto",
    )

    ica.fit(epochs, decim=2)
    print(f"  ICA fitted: {ica.n_components_} components")

    # ICLabel for automatic classification
    ic_labels = label_components(epochs, ica, method="iclabel")
    labels = ic_labels["labels"]
    probs  = ic_labels["y_pred_proba"]

    print("\n  Component classification:")

    brain_ics = []
    exclude_ics = []

    for i, (label, prob_vec) in enumerate(zip(labels, probs)):
        prob = prob_vec.max()
        print(f"    IC{i:03d}  {label:<20} p={prob:.2f}")

        if label == BRAIN_LABEL and prob >= BRAIN_THRESHOLD:
            brain_ics.append(i)
        else:
            exclude_ics.append(i)

    print(f"\n  Keeping {len(brain_ics)} brain ICs: {brain_ics}")
    print(f"  Excluding {len(exclude_ics)} non-brain ICs")

    ica.exclude = exclude_ics

    # Apply ICA to raw data
    raw_clean = ica.apply(raw.copy())

    # Save 
    ica.save(OUTPUT_DIR / f"sub-{subject}_ica.fif", overwrite=True)
    raw_clean.save(OUTPUT_DIR / f"sub-{subject}_desc-clean_raw.fif", overwrite=True)

    print(f"  Saved ICA solution → sub-{subject}_ica.fif")
    print(f"  Saved clean raw    → sub-{subject}_desc-clean_raw.fif")

print("\nICA pipeline complete.")