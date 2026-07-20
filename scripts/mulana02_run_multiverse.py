"""
betaGaitMultiverse — run all universes.

Requires mulana01_create_multiverse.py to have been run first so that
the universe scripts exist. Runs all universes sequentially (parallel=1).
"""

from comet.multiverse import Multiverse
from src.config import MULTIVERSE_NAME, DIR_MULTIVERSE_COMET

# path= must match mulana01's -- see src.config.DIR_MULTIVERSE_COMET.
mverse = Multiverse(name=MULTIVERSE_NAME, path=str(DIR_MULTIVERSE_COMET))
mverse.run(parallel=1)
