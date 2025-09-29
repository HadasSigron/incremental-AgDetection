# core/types/runner_result.py
from typing import Dict, TypedDict, Any


class RunnerResult(TypedDict):
    """
    The standard result that every Runner must return.
    - metrics: performance metrics (float)
    - artifacts: file paths/URIs of produced artifacts (str)
    """
    metrics: Dict[str, float]
    artifacts: Dict[str, str]


def ensure_runner_result(obj: Any) -> RunnerResult:
    """
    Ensures the result structure is valid, and converts 'tricky' types 
    (like numpy.float32) into float/str.
    Raises ValueError if the structure is invalid.
    """
    if not isinstance(obj, dict) or "metrics" not in obj or "artifacts" not in obj:
        raise ValueError("Runner must return {'metrics': {...}, 'artifacts': {...}}")

    metrics = obj["metrics"]
    artifacts = obj["artifacts"]

    if not isinstance(metrics, dict) or not isinstance(artifacts, dict):
        raise ValueError("metrics/artifacts must be dicts")

    # safe conversions
    safe_metrics = {str(k): float(v) for k, v in metrics.items()}
    safe_artifacts = {str(k): str(v) for k, v in artifacts.items()}

    return {"metrics": safe_metrics, "artifacts": safe_artifacts}
