import mne
import numpy as np
from autoreject import AutoReject
from src.paths import get_dataset_dirs

DATASET = "splitbelt"
dirs = get_dataset_dirs(DATASET)

INPUT_DIR  = dirs["sigclean"]
OUTPUT_DIR = dirs["preica"]

BAD_CHAN_THRESHOLD = 3.0  
EPOCH_DURATION = 2.0 # seconds

SUBJECTS = ["S18"]

for subject in SUBJECTS:
    print(f"\nProcessing {subject}")

    raw = mne.io.read_raw_fif(
        INPUT_DIR / f"sub-{subject}_clean_raw.fif", preload=True
    )

    # Bad channel detection and interpolation
    data = raw.get_data()
    var  = np.var(data, axis=1)
    z    = (var - var.mean()) / var.std()
    bads = [raw.ch_names[i] for i in np.where(np.abs(z) > BAD_CHAN_THRESHOLD)[0]]
    raw.info["bads"] = bads
    print(f"  Bad channels ({len(bads)}): {bads}")

    if bads:
        raw.interpolate_bads(reset_bads=True)
        print(f"  Interpolated {len(bads)} channels")

    # Average re-reference 
    raw.set_eeg_reference("average", projection=False)
    print("  Re-referenced to average")


    # Epoch for AutoReject 
    events = mne.make_fixed_length_events(raw, id=1, duration=EPOCH_DURATION)
    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=EPOCH_DURATION,
        baseline=None, preload=True
    )
    print(f"  Created {len(epochs)} epochs of {EPOCH_DURATION} s")

    # AutoReject for bad epoch detection 
    ar = AutoReject(n_interpolate=[1, 2, 4], random_state=42, n_jobs=1, verbose=False)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)
    n_bad = reject_log.bad_epochs.sum()
    print(f"  AutoReject: {n_bad}/{len(epochs)} epochs rejected "
          f"({100*n_bad/len(epochs):.0f}%)")

    if n_bad / len(epochs) > 0.30:
        print("  WARNING: >30% epochs rejected — check ASR cutoff or bad channel threshold")

    # Save pre-ICA epochs
    out = OUTPUT_DIR / f"sub-{subject}_preica_clean_epo.fif"
    epochs_clean.save(out, overwrite=True)
    print(f"  Saved → {out.name}")