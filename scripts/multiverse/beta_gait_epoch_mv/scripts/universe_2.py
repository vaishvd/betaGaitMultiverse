import comet
from src.multiverse import run_pipeline  

epoch_length = 1.0

try:
    result = run_pipeline(epoch_length)
    result["epoch_length"] = epoch_length
    result["status"] = "success"

except Exception as e:
    result = {
        "epoch_length": epoch_length,
        "status": "fail",
        "error": str(e)
    }

comet.utils.save_universe_results(result)