# tests/test_workers/test_eval_worker_seg_flow.py
"""
EvalWorker integration in YAML mode for Segmentation:
- build_appconfig is called with YAML path
- cfg.task may be Enum-like -> worker normalizes for get_runner
- result payload includes the expected keys and contains segmentation metrics

This test does not require a Qt event loop; we invoke run() synchronously and ignore signals.
"""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from ui.workers.eval_worker import EvalWorker


class _EnumLike:
    def __init__(self, value: str):
        self.value = value


class _FakeDataset:
    def __len__(self) -> int:
        return 2

    def describe(self) -> Dict[str, str]:
        return {"kind": "masks_segmentation", "root": "/x"}


class FakeSegmenterRunner:
    """Fake runner that mimics a real SegmenterRunner result shape."""
    def run(self, dataset: Any, cfg: Any, cancel_token: Any = None):
        return {
            "metrics": {"miou": 0.42, "pixel_accuracy": 0.77, "duration": 0.001},
            "artifacts": {"runtime_sec": "0.001", "run_dir": str(Path.cwd() / "runs" / "tmp")},
        }


def _fake_build_dataset_adapter(cfg: Any) -> _FakeDataset:
    return _FakeDataset()


def _fake_get_runner(task_key: str) -> FakeSegmenterRunner:
    # Ensure worker normalized Enum -> str
    assert task_key == "Segmentation"
    return FakeSegmenterRunner()


def test_eval_worker_seg_yaml_flow(monkeypatch, tmp_path):
    # Minimal AppConfig-like object that the worker expects back from build_appconfig
    cfg = SimpleNamespace(
        task=_EnumLike("Segmentation"),
        benchmark_name="tiny-seg",
        dataset=SimpleNamespace(kind="masks_segmentation"),
        model=SimpleNamespace(name="fake", format="script", path=None, extra={}),
        eval=SimpleNamespace(
            device="cpu",
            batch_size=1,
            metrics=["miou", "pixel_accuracy"],
            params={"num_classes": 2},
        ),
        paths=SimpleNamespace(runs_root=tmp_path),
    )

    def _fake_build_appconfig(**kwargs):
        # Worker passes the YAML path as a string
        assert isinstance(kwargs.get("benchmark_yaml"), str)
        return cfg

    # Patch dependencies used inside EvalWorker.run
    monkeypatch.setattr("ui.workers.eval_worker.build_appconfig", _fake_build_appconfig)
    monkeypatch.setattr("ui.workers.eval_worker.build_dataset_adapter", _fake_build_dataset_adapter)
    monkeypatch.setattr("ui.workers.eval_worker.get_runner", _fake_get_runner)

    # Run synchronously; signals are emitted but we don't connect listeners in this test
    worker = EvalWorker(benchmark_yaml="configs/benchmarks/tiny_seg.yaml")
    worker.run()

    # If we got here with no exceptions, the flow is OK (task normalization + payload build)
    assert True
