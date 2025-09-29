# -*- coding: utf-8 -*-
# Check that EvalWorker emits 'finished' and returns a result dict with 'metrics.duration'.
# Speed up by monkeypatching sleep.

import sys
from pathlib import Path
import pytest  # noqa: F401  (kept for pytest collection)

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ui.workers.eval_worker import EvalWorker
from core.background.cancel_token import CancelToken


def test_eval_worker_signals(qtbot, monkeypatch):
    # Speed up: patch time.sleep to no-op
    import time
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    dataset = {"benchmark_id": "bench-1", "domain": "P2"}
    config = {"taskType": "Detection", "device": "CPU", "batch_size": 4, "metric_preset": "default"}
    cancel = CancelToken()

    worker = EvalWorker(dataset, config, cancel)

    # Start the worker and wait for the 'finished' signal properly (as a context manager)
    with qtbot.waitSignal(worker.signals.finished, timeout=3000, raising=True) as finished_spy:
        worker.start()

    result = finished_spy.args[0]  # the dict emitted

    assert isinstance(result, dict)
    assert "metrics" in result
    assert "duration" in result["metrics"]
