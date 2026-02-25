import openneuro as on
from utils.config import dir_data

on.download(dataset='ds004475', target_dir= dir_data)