import mne
import numpy as np
from src.config import DIR_SIGCLEAN   
from asrpy import ASR

INPUT_DIR = DIR_SIGCLEAN
raw_file = INPUT_DIR / "sub-S18_clean_raw.fif"

# Get raw data as numpy array
raw = mne.io.read_raw_fif(raw_file, preload=True)   
data = raw.get_data()  # shape: (n_channels, n_samples)
sfreq = raw.info['sfreq']

# define the ASR object with standard parameters
asr = ASR(sfreq=sfreq, cutoff=20)

# fit it to the first 100 seconds of the recording
asr.fit(raw.copy().crop(tmax=100))

# create a copy of our data and fill in the ASR cleaned data

raw_asr = asr.transform(raw)

# create epochs from the cleaned data
epochs_asr = mne.Epochs(raw_asr, baseline=None,
                        reject=None, verbose=False, detrend=0, preload=True)