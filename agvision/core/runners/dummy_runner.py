from __future__ import annotations
import time
from typing import Any, Dict

# Try to inherit from the real base Runner if it exists, otherwise provide a stub.
try:
    from core.runners.base import Runner as _BaseRunner  # ABC, if present
except Exception:
    class _BaseRunner:
        """Fallback base with the expected interface."""
        def run(self, dataset, config, cancel_token):
            raise NotImplementedError

from core.types.runner_result import RunnerResult, ensure_runner_result


class DummyRunner(_BaseRunner):
    """
    Minimal runner used for tests.
    It simulates work, respects a cancel token,
    and returns a valid RunnerResult (dict with 'metrics' and 'artifacts').
    """

    def __init__(self, **kwargs: Any) -> None:
        # Accept arbitrary kwargs to match potential factory signatures
        self._kwargs = kwargs

    def run(self, dataset: Any, config: Any, cancel_token: Any = None) -> RunnerResult:
        steps = 10
        start = time.time()

        for _ in range(steps):
            # Cooperative cancel support
            if cancel_token is not None and getattr(cancel_token, "is_set", lambda: False)():
                break
            time.sleep(0.01)  # simulate some workload

        duration = time.time() - start

        # Build the result structure expected by tests and the UI
        result: RunnerResult = {
            "metrics": {
                "duration_s": round(duration, 3),
                "Accuracy": 0.5,
                "mAP@0.5": 0.1,
                "mIoU": 0.2,
            },
            "artifacts": {
                "predictions_path": None,
            },
        }
        # Ensure result has correct types and structure
        return ensure_runner_result(result)
