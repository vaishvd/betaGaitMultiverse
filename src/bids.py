from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids


def get_subjects(data_root):
    """Return sorted list of subject IDs."""
    return sorted(
        d.name.replace("sub-", "")
        for d in Path(data_root).iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )


def load_raw_bids(subject, task, datatype, root):
    """Load BIDS raw data."""
    bids_path = BIDSPath(
        subject=subject,
        task=task,
        datatype=datatype,
        root=root,
    )

    raw = read_raw_bids(bids_path)

    return raw