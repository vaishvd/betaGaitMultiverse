"""
betaGaitMultiverse — run all universes.

Requires mulana01_create_multiverse.py to have been run first so that
the universe scripts exist. Runs all universes sequentially (parallel=1).
"""

from comet.multiverse import Multiverse
from src.config import MULTIVERSE_NAME

mverse = Multiverse(name=MULTIVERSE_NAME)
mverse.run(parallel=1)
