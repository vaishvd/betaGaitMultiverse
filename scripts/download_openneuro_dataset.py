import openneuro as on
from src.config import DIR_RAWDATA

on.download(dataset='ds004475', target_dir=DIR_RAWDATA, exclude=['**/anat', '**/headmodel'])