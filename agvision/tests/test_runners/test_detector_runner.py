from __future__ import annotations
import json
from pathlib import Path
import pytest
from PIL import Image

# Import works whether tests run from repo root or from inside agvision/
try:
    from core.runners.detector_runner import DetectorRunner
    from core.datasets.coco_detection import COCODetectionAdapter
except ModuleNotFoundError:
    from core.runners.detector_runner import DetectorRunner
    from core.datasets.coco_detection import COCODetectionAdapter


def norm_result(res):
    if isinstance(res, tuple) and len(res) == 2:
        m, a = res
        return {"metrics": m, "artifacts": a}
    if isinstance(res, dict) and "metrics" in res and "artifacts" in res:
        return res
    raise AssertionError("Unexpected runner result; expected tuple or dict with 'metrics'/'artifacts'.")


def _make_mini_coco(root: Path) -> tuple[str, str]:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_json = root / "annotations.json"

    ann = {
        "info": {"description": "mini-coco for tests"},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "000001.jpg", "width": 640, "height": 480, "license": 0},
            {"id": 2, "file_name": "000002.jpg", "width": 640, "height": 480, "license": 0},
        ],
        "categories": [
            {"id": 1, "name": "weed"},
            {"id": 2, "name": "plant"},
            {"id": 3, "name": "fruit"},
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "category_id": 1,
             "bbox": [50, 60, 80, 40], "area": 3200, "iscrowd": 0}
        ],
    }
    ann_json.write_text(json.dumps(ann), encoding="utf-8")

    def _write_jpeg(p: Path, size=(640, 480), color=(0, 0, 0)):
        img = Image.new("RGB", size, color)
        img.save(p, format="JPEG", quality=85)

    _write_jpeg(images_dir / "000001.jpg", size=(640, 480), color=(0, 0, 0))
    _write_jpeg(images_dir / "000002.jpg", size=(640, 480), color=(10, 10, 10))

    return str(images_dir), str(ann_json)

class _CancelNever:
    def is_set(self) -> bool:  # Runner checks this flag during the loop
        return False


class _CancelImmediately:
    def is_set(self) -> bool:
        return True


def test_simulate_smoke(tmp_path: Path):
    images_dir, ann_json = _make_mini_coco(tmp_path / "mini_coco")
    ds = COCODetectionAdapter(images_dir, ann_json)

    cfg = {
        "domain": "P3",
        "benchmark_id": "bench01",
        "simulate": True,
        "batch_size": 1,
        "out_dir": None,  # let runner create runs/... structure
    }

    res = DetectorRunner().run(ds, cfg, _CancelNever())
    result = norm_result(res)
    artifacts = result["artifacts"]
    metrics = result["metrics"]

    assert "predictions_json" in artifacts
    assert Path(artifacts["predictions_json"]).exists()
    for k in ("mAP@0.5:0.95", "AP@0.5", "AP@0.75", "AR_all"):
        assert k in metrics


def test_eval_only(tmp_path: Path):
    images_dir, ann_json = _make_mini_coco(tmp_path / "mini_coco")
    ds = COCODetectionAdapter(images_dir, ann_json)

    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [50, 60, 80, 40], "score": 0.9},
        {"image_id": 2, "category_id": 2, "bbox": [30, 40, 50, 30], "score": 0.7},
    ]
    preds_path = tmp_path / "predictions.json"
    preds_path.write_text(json.dumps(preds), encoding="utf-8")

    cfg = {
        "domain": "P3",
        "benchmark_id": "bench01",
        "predictions_json": str(preds_path),
        "out_dir": str(tmp_path / "run_out"),
    }

    res = DetectorRunner().run(ds, cfg, _CancelNever())
    result = norm_result(res)
    artifacts = result["artifacts"]
    metrics = result["metrics"]

    out_p = Path(artifacts["predictions_json"])
    assert out_p.exists()
    # ואפשר גם לוודא שהתוכן באמת זהה
    
    orig = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    copy = json.loads(out_p.read_text(encoding="utf-8"))
    assert copy == orig

    for k in ("mAP@0.5:0.95", "AP@0.5", "AP@0.75", "AR_all"):
        assert k in metrics


def test_cancel_early(tmp_path: Path):
    images_dir, ann_json = _make_mini_coco(tmp_path / "mini_coco")
    ds = COCODetectionAdapter(images_dir, ann_json)
    cfg = {"domain": "P3", "benchmark_id": "bench01", "simulate": True, "batch_size": 1}

    res = DetectorRunner().run(ds, cfg, _CancelImmediately())
    result = norm_result(res)
    artifacts = result["artifacts"]
    metrics = result["metrics"]

    assert "predictions_json" in artifacts
    for k in ("mAP@0.5:0.95", "AP@0.5", "AP@0.75", "AR_all"):
        assert k in metrics


@pytest.mark.skipif("YOLO_WEIGHTS" not in __import__("os").environ,
                    reason="Set YOLO_WEIGHTS env var to run this test")
def test_yolo_optional(tmp_path: Path):
    import os
    weights = os.environ["YOLO_WEIGHTS"]

    images_dir, ann_json = _make_mini_coco(tmp_path / "mini_coco")
    ds = COCODetectionAdapter(images_dir, ann_json)

    cfg = {
        "domain": "P3",
        "benchmark_id": "bench01",
        "model_path": weights,     # e.g., /weights/yolov8n.pt
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "imgsz": 640,
        "batch_size": 1,
        "out_dir": None,
    }

    res = DetectorRunner().run(ds, cfg, _CancelNever())
    result = norm_result(res)
    artifacts = result["artifacts"]
    metrics = result["metrics"]

    assert Path(artifacts["predictions_json"]).exists()
    for k in ("mAP@0.5:0.95", "AP@0.5", "AP@0.75", "AR_all"):
        assert k in metrics
