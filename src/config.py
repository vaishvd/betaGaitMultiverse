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
DIR_DATA = define_dir(dir_proj, "data")
DIR_RAWDATA = define_dir(DIR_DATA, "raw")
