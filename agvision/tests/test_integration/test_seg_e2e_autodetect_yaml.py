# tests/test_integration/test_seg_e2e_autodetect_yaml.py
"""
End-to-end smoke for Segmentation:
- Simulate "model upload" + autodetect results (format/arch + thresholds)
- Build a minimal AppConfig-like object (as if YAML+autodetect were merged)
- Run SegmenterRunner and assert:
  * loader args honor precedence: model.* > eval.params.*
  * requested metrics are computed exactly
  * predictions.json is created and has the expected schema

Note: We no longer require run_config.json here because, in the current
architecture, it is written by the EvalWorker layer, not by the Runner itself.
"""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List
import json

import numpy as np
from PIL import Image
import pytest

from core.runners.segmenter_runner import SegmenterRunner


# ------------------------------- fakes ----------------------------------------


class _TinySegDataset:
    """2 samples with tiny masks on disk; adapter-like API (iter_samples)."""
    def __init__(self, root: Path) -> None:
        self.root = root
        self.imgs = [root / "img0.png", root / "img1.png"]
        self.gts = [root / "gt0.png", root / "gt1.png"]

        root.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((2, 2), dtype="uint8")).save(self.imgs[0])
        Image.fromarray(np.ones((2, 2), dtype="uint8") * 127).save(self.imgs[1])
        # masks: classes {0,1}
        Image.fromarray(np.array([[0, 1], [1, 1]], dtype="uint8")).save(self.gts[0])
        Image.fromarray(np.array([[1, 0], [0, 0]], dtype="uint8")).save(self.gts[1])

    def __len__(self) -> int:
        return 2

    def iter_samples(self) -> Iterable[Dict[str, Any]]:
        for i, (ip, gp) in enumerate(zip(self.imgs, self.gts)):
            yield {"id": i, "path": ip.as_posix(), "mask_path": gp.as_posix()}


class _FakeSegModel:
    """Returns a plausible predicted mask; stands in for real inference."""
    def __init__(self, flip: bool = False) -> None:
        self.flip = flip

    def predict(self, path: str):
        # Deterministic per file name
        p = Path(path).name
        base = np.array([[0, 1], [1, 1]], dtype="uint8")
        if "img1" in p:
            base = np.array([[1, 0], [0, 0]], dtype="uint8")
        return 1 - base if self.flip else base


# ------------------------------- helpers --------------------------------------


def _cfg_like(tmp: Path, yaml_cfg: Dict[str, Any]) -> Any:
    """
    Minimal AppConfig-ish object: only attributes used by SegmenterRunner.
    This imitates a "validated" config the UI would pass after YAML + autodetect merge.
    """
    return SimpleNamespace(
        task="Segmentation",
        domain=SimpleNamespace(value=yaml_cfg.get("domain", "weeds")),
        benchmark_name=yaml_cfg.get("benchmark_name", "tiny-seg"),
        paths=SimpleNamespace(runs_root=tmp),
        dataset=SimpleNamespace(kind=yaml_cfg["dataset"]["kind"]),
        model=SimpleNamespace(
            name=yaml_cfg["model"]["name"],
            format=yaml_cfg["model"]["format"],
            path=yaml_cfg["model"].get("path"),
            imgsz=yaml_cfg["model"].get("imgsz", None),
            # autodetect or user-provided model-level thresholds:
            conf_threshold=yaml_cfg["model"].get("conf_threshold", None),
            iou_threshold=yaml_cfg["model"].get("iou_threshold", None),
            mask_threshold=yaml_cfg["model"].get("mask_threshold", None),
            extra=yaml_cfg["model"].get("extra", {}),
        ),
        eval=SimpleNamespace(
            device=yaml_cfg["eval"]["device"],
            batch_size=yaml_cfg["eval"]["batch_size"],
            metrics=list(yaml_cfg["eval"]["metrics"]),
            params=dict(yaml_cfg["eval"].get("params", {})),
        ),
    )


# ------------------------------- test -----------------------------------------


def test_segmentation_e2e_autodetect_yaml(tmp_path, monkeypatch):
    """
    Flow:
      * "Uploaded" weights -> autodetect suggests format/arch + thresholds
      * YAML refers to Segmentation + metrics ["miou","pixel_accuracy"]
      * Runner uses fake model; checks that params precedence works:
        model.* overrides eval.params.*
    """
    # --- 1) prepare tiny dataset
    ds = _TinySegDataset(tmp_path / "data")

    # --- 2) pretend we uploaded a file "model.onnx" and autodetected it
    uploaded_weights = (tmp_path / "weights" / "model.onnx")
    uploaded_weights.parent.mkdir(parents=True, exist_ok=True)
    uploaded_weights.write_bytes(b"dummy")  # file exists; contents irrelevant

    autodetect = {
        "format": "onnx",
        "arch": "generic-seg",
        # autodetect proposes defaults:
        "conf_threshold": 0.35,
        "iou_threshold": 0.55,
        "mask_threshold": 0.40,
    }

    # --- 3) YAML that matches Segmentation task
    yaml_dict = {
        "task": "Segmentation",
        "domain": "weeds",
        "benchmark_name": "tiny-seg",
        "dataset": {"kind": "masks_segmentation"},
        "model": {
            "name": "uploaded_generic_seg",
            "format": autodetect["format"],     # taken from autodetect
            "path": uploaded_weights.as_posix(),
            # NOTE: model-level overrides beat eval.params (asserted later)
            "conf_threshold": 0.60,             # override > autodetect & > eval.params
        },
        "eval": {
            "device": "cpu",
            "batch_size": 1,
            "metrics": ["miou", "pixel_accuracy"],
            "params": {
                "num_classes": 2,
                "conf_threshold": 0.10,        # should be ignored by runner (model.* wins)
                "iou_threshold": 0.50,
                "mask_threshold": 0.50,
            },
        },
    }

    # --- 4) make sure SegmenterRunner will call our fake loader/model
    captured_loader_args: Dict[str, Any] = {}

    def _fake_load_segmenter(args: Dict[str, Any]) -> _FakeSegModel:
        # Capture what the runner passes after merging model.* and eval.params.*
        captured_loader_args.update(args)
        return _FakeSegModel(flip=False)

    monkeypatch.setattr("core.runners.segmenter_runner.load_segmenter", _fake_load_segmenter)

    # --- 5) build minimal AppConfig-like instance
    cfg = _cfg_like(tmp_path, yaml_dict)

    # --- 6) run
    r = SegmenterRunner()
    out = r.run(ds, cfg, cancel_token=None)

    # --- 7) asserts

    # a) loader received precedence-resolved thresholds
    assert captured_loader_args["model_path"].endswith("model.onnx")
    # model.conf_threshold (0.60) overrides eval.params (0.10) and autodetect
    assert pytest.approx(captured_loader_args["conf_threshold"], rel=0, abs=1e-9) == 0.60
    # iou/mask thresholds came from eval.params (since model.* didn't override them)
    assert pytest.approx(captured_loader_args["iou_threshold"], rel=0, abs=1e-9) == 0.50
    assert pytest.approx(captured_loader_args["mask_threshold"], rel=0, abs=1e-9) == 0.50
    assert captured_loader_args["device"] == "cpu"

    # b) metrics are exactly the requested set
    assert set(out["metrics"].keys()) == {"miou", "pixel_accuracy"}

    # c) artifacts exist (predictions.json only; run_config.json is written by the Worker layer)
    run_dir = Path(out["artifacts"]["run_dir"])
    assert (run_dir / "predictions.json").exists(), "predictions.json must be saved"

    # d) predictions.json has the expected schema (id, path, mask)
    preds = json.loads((run_dir / "predictions.json").read_text("utf-8"))
    assert isinstance(preds, list) and len(preds) == len(ds)
    assert {"id", "path", "mask"}.issubset(preds[0].keys())
