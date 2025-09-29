# tests/test_runners/test_classifier_runner.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, List
import json
import math
import pytest

from core.runners.classifier_runner import ClassifierRunner


# ---------- Helpers ----------

def norm_result(res) -> Dict[str, Dict]:
    """
    Normalize runner result to a dict with keys 'metrics' and 'artifacts'.
    Supports both tuple (metrics, artifacts) and dict {'metrics','artifacts'}.
    """
    if isinstance(res, tuple) and len(res) == 2:
        metrics, artifacts = res
        return {"metrics": metrics, "artifacts": artifacts}
    if isinstance(res, dict) and "metrics" in res and "artifacts" in res:
        return res
    raise AssertionError("Unexpected runner result type. Expected tuple or dict with 'metrics'/'artifacts'.")


def read_predictions(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


class ToyClassificationDataset:
    """
    Minimal in-memory classification adapter for tests.

    Exposes:
      - iter_samples() -> iterable of dicts with keys {id, path, label}
      - class_names() -> list[str]
      - num_classes() -> int
    """
    def __init__(self, tmp_dir: Path, labels: List[int], n_classes: int):
        self._samples = []
        self._class_names = [f"class_{i}" for i in range(n_classes)]
        for i, lab in enumerate(labels, start=1):
            self._samples.append({
                "id": i,
                "path": str(tmp_dir / f"img_{i}.jpg"),
                "label": int(lab),
            })

    def iter_samples(self) -> Iterable[Dict[str, Any]]:
        for s in self._samples:
            yield s

    def class_names(self) -> List[str]:
        return list(self._class_names)

    def num_classes(self) -> int:
        return len(self._class_names)

    def size(self) -> int:
        return len(self._samples)


# ---------- Tests ----------

def test_simulate_run_artifacts_and_metrics(tmp_path: Path):
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    ds = ToyClassificationDataset(tmp_path, labels, n_classes=3)

    cfg = {
        "domain": "P2",
        "benchmark_id": "toy",
        "simulate": True,
        "batch_size": 3,
        "imgsz": 224,
        "device": "cpu",
        "out_dir": str(tmp_path / "out_sim"),
    }

    class Cancel:
        def is_set(self): return False

    res = ClassifierRunner().run(ds, cfg, Cancel())
    result = norm_result(res)

    a = result["artifacts"]
    pred_p = Path(a["predictions_json"])
    cm_p = Path(a["confusion_matrix_png"])
    assert pred_p.exists(), "predictions.json not created"
    assert cm_p.exists(), "confusion_matrix.png not created"

    preds = read_predictions(pred_p)
    assert len(preds) == ds.size()

    m = result["metrics"]
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in m
        assert isinstance(m[key], float)
        assert 0.0 <= m[key] <= 1.0


def test_eval_only_known_values(tmp_path: Path):
    class DS:
        def class_names(self): return ["class_0", "class_1", "class_2"]

    preds = [
        {"id": 1, "path": "x/1.jpg", "label": 0, "pred": 0},
        {"id": 2, "path": "x/2.jpg", "label": 1, "pred": 1},
        {"id": 3, "path": "x/3.jpg", "label": 2, "pred": 0},
        {"id": 4, "path": "x/4.jpg", "label": 2, "pred": 2},
    ]
    preds_path = tmp_path / "eval_only_preds.json"
    preds_path.write_text(json.dumps(preds), encoding="utf-8")

    cfg = {
        "domain": "P2",
        "benchmark_id": "toy_eval_only",
        "predictions_json": str(preds_path),
        "out_dir": str(tmp_path / "out_eval"),
        "average": "macro",
    }

    class Cancel:
        def is_set(self): return False

    res = ClassifierRunner().run(DS(), cfg, Cancel())
    result = norm_result(res)
    m = result["metrics"]

    assert math.isclose(m["accuracy"], 0.75, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(m["precision"], 5.0 / 6.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(m["recall"],    5.0 / 6.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(m["f1"],        7.0 / 9.0, rel_tol=1e-6, abs_tol=1e-6)


def test_cancel_midway_skips_partial_batch(tmp_path: Path):
    labels = list(range(7))
    ds = ToyClassificationDataset(tmp_path, labels, n_classes=5)

    out_dir = tmp_path / "out_cancel"
    bs = 3

    class Cancel:
        def __init__(self, allowed_checks: int):
            self.calls = 0
            self.allowed = allowed_checks
        def is_set(self):
            self.calls += 1
            return self.calls > self.allowed

    cfg = {
        "domain": "P2",
        "benchmark_id": "toy_cancel",
        "simulate": True,
        "batch_size": bs,
        "out_dir": str(out_dir),
    }

    res = ClassifierRunner().run(ds, cfg, Cancel(allowed_checks=5))
    result = norm_result(res)
    pred_p = Path(result["artifacts"]["predictions_json"])
    preds = read_predictions(pred_p)

    assert len(preds) > 0
    assert len(preds) < ds.size()
    assert len(preds) % bs == 0


def test_average_modes_smoke(tmp_path: Path):
    labels = [0, 1, 0, 1, 0, 1]
    ds = ToyClassificationDataset(tmp_path, labels, n_classes=2)

    class Cancel:
        def is_set(self): return False

    for avg in ("macro", "micro", "weighted"):
        cfg = {
            "domain": "P2",
            "benchmark_id": f"toy_{avg}",
            "simulate": True,
            "batch_size": 2,
            "average": avg,
            "out_dir": str(tmp_path / f"out_{avg}"),
        }
        res = ClassifierRunner().run(ds, cfg, Cancel())
        result = norm_result(res)
        m = result["metrics"]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in m
            assert isinstance(m[key], float)
