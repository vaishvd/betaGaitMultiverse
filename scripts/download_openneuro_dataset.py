import openneuro as on
from src.config import DIR_DATA

on.download(dataset='ds004475', target_dir=DIR_DATA, exclude=['**/anat', '**/headmodel'])