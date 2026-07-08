"""
betaGaitMultiverse — multiverse pipeline entry point.

Runs all three multiverse stages in sequence:
  1. mulana01_create_multiverse.py  — clear old state, define decisions, create universes
  2. mulana02_run_multiverse.py     — run all universes
  3. mulana03_visualize_multiverse.py — specification curve and density plots
"""

import subprocess
import sys
import time

from src.config import DIR_SCRIPTS

STAGES = [
    (1, "mulana01_create_multiverse.py",  "clear state + create universes"),
    (2, "mulana02_run_multiverse.py",     "run all universes"),
    (3, "mulana03_visualize_multiverse.py", "specification curve + density plot"),
]

print("\nbetaGaitMultiverse — multiverse pipeline")

results = []
t_total = time.time()

for stage_num, script_name, description in STAGES:
    script = DIR_SCRIPTS / script_name
    print(f"\n{'='*60}")
    print(f"Stage {stage_num}/3 — {description}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(DIR_SCRIPTS.parent)
    )
    elapsed = time.time() - t0
    ok = result.returncode == 0

    status = "OK" if ok else f"FAILED (code {result.returncode})"
    print(f"\n[{status}] Stage {stage_num} in {elapsed:.1f}s")
    results.append((stage_num, description, ok))

print(f"\n{'='*60}")
print(f"Multiverse pipeline complete in {(time.time()-t_total)/60:.1f} min")
print(f"{'='*60}")
for stage_num, description, ok in results:
    print(f"  [{'OK  ' if ok else 'FAIL'}] Stage {stage_num}: {description}")
