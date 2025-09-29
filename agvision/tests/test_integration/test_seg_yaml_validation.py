"""
Validation checks for segmentation-like configs.
We emulate a minimal YAML and ensure mismatches or missing keys are flagged.
If your real schema.py raises specific exceptions, adapt the assertion accordingly.
"""
from __future__ import annotations
from types import SimpleNamespace
import pytest

# If you have a real validator (core.config.schema), import and use it.
# Here we show a minimal expectation: the runner requires task + dataset.kind.
from core.runners.segmenter_runner import SegmenterRunner

def _cfg_like(task: str | object, dataset_kind: str | None):
    return SimpleNamespace(
        task=task,
        domain=SimpleNamespace(value="weeds"),
        benchmark_name="x",
        paths=SimpleNamespace(runs_root="runs"),
        dataset=SimpleNamespace(kind=dataset_kind) if dataset_kind else None,
        model=SimpleNamespace(name="fake", format="script", path=None, extra={}),
        eval=SimpleNamespace(device="cpu", batch_size=1, metrics=["miou"], params={"num_classes": 2}),
    )

def test_yaml_validation__task_mismatch():
    cfg = _cfg_like(task="Classification", dataset_kind="masks_segmentation")
    with pytest.raises(Exception):
        # SegmenterRunner expects segmentation dataset semantics somewhere in the flow.
        SegmenterRunner().run(dataset=[], cfg=cfg, cancel_token=None)  # type: ignore[arg-type]

def test_yaml_validation__missing_dataset_kind():
    cfg = _cfg_like(task="Segmentation", dataset_kind=None)
    with pytest.raises(Exception):
        SegmenterRunner().run(dataset=[], cfg=cfg, cancel_token=None)  # type: ignore[arg-type]
