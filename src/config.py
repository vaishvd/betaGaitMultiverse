from pathlib import Path

def define_dir(root, *names):
    """Creates a directory and ensures it exists."""
    path = root
    for name in names:
        path = path / name 
    path.mkdir(parents=True, exist_ok=True)
    return path

# Get the root directory of the repository (parent of 'src')
dir_proj = Path(__file__).resolve().parents[1]

# Define paths for data directories and work notebooks
DIR_DATA = define_dir(dir_proj, "data") # Data folder
DIR_RAWDATA = define_dir(DIR_DATA, "d00_raw") # Raw datasets
DIR_MONTAGE = define_dir(DIR_DATA, "d00_montage") # Raw datasets with montage set
DIR_SEG = define_dir(DIR_DATA, "d01_segmented") # Datasets segmented with events of interest
DIR_SIGCLEAN = define_dir(DIR_DATA, "d02_sigclean") # Data after signal cleaning
DIR_PREICA = define_dir(DIR_DATA, "d03_preica") # Data before ICA
DIR_ICA = define_dir(DIR_DATA, "d04_ica") # Data after ICA
DIR_GAIT = define_dir(DIR_DATA, "d05_gaitcycles") # Extracted and time-normalized gait cycles
DIR_TFR = define_dir(DIR_DATA, "d06_tfr")
DIR_ERSP = define_dir(DIR_DATA, "d07_ersp")
DIR_PLOT = define_dir(DIR_DATA, "d09_plotgaitbeta")