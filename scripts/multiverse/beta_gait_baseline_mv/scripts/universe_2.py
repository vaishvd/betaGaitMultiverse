import sys
from pathlib import Path

# Make sure src is importable
sys.path.append(str(Path(__file__).resolve().parents[2]))

import comet
from src.multiverse import run_pipeline

baseline = "global"

try:
    result = run_pipeline(baseline=baseline)

    result["baseline"] = baseline
    result["status"] = "success"

except Exception as e:
    result = {
        "baseline": baseline,
        "status": "fail",
        "error": str(e)
    }

comet.utils.save_universe_results(result)