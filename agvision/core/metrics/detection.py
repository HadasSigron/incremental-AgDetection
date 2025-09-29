from __future__ import annotations
import json
import os

from pycocotools.cocoeval import COCOeval
# agvision/core/metrics/detection.py
from typing import Dict, Iterable, List

# assume evaluate_coco(...) already exists in this module

def compute_metrics_from_config(
    dataset: any,
    preds_json_path: str,
    img_ids: List[int],
    *,
    requested: Iterable[str],
    strict: bool = True,
) -> Dict[str, float]:
    """
    Compute detection metrics strictly as requested by config.

    Implementation:
      - Uses COCO evaluation via evaluate_coco(dataset.coco_gt(), preds_json_path, img_ids)
      - Filters the resulting metrics dict to the requested names (case-insensitive).
      - If 'strict' is True and a requested name is not available, raises KeyError.

    Notes:
      - This function is backend-agnostic from the runner perspective. If you later add
        other backends (e.g., VOC), you can dispatch here instead of in the runner,
        keeping the "metrics computation lives in metrics package" principle.
    """
    metrics_full: Dict[str, float] = evaluate_coco(dataset.coco_gt(), preds_json_path, img_ids)

    # Build case-insensitive name map
    key_map = {k.lower(): k for k in metrics_full.keys()}
    req = [str(n).lower() for n in requested]

    result: Dict[str, float] = {}
    unknown: List[str] = []
    for nm in req:
        if nm in key_map:
            orig = key_map[nm]
            result[orig] = metrics_full[orig]
        else:
            unknown.append(nm)

    if unknown and strict:
        raise KeyError(
            f"Unknown detection metric names requested: {unknown}. "
            f"Available: {list(metrics_full.keys())}"
        )

    # In strict mode, return {} if nothing matched; otherwise (non-strict) fall back to full set
    return result if result else ({} if strict else metrics_full)



def _zero_metrics() -> Dict[str, float]:
    return {
        "mAP@0.5:0.95": 0.0,
        "AP@0.5": 0.0,
        "AP@0.75": 0.0,
        "AR_all": 0.0,
    }

def evaluate_coco(coco_gt, predictions_json: str, img_ids: List[int] | None = None) -> Dict[str, float]:
    """
    Evaluate COCO metrics (bbox). Robust to:
    - missing 'info'/'licenses' in GT
    - empty predictions (skip loadRes; return zero metrics)
    """
    # If predictions file is missing or empty list, return zeros.
    if not os.path.exists(predictions_json):
        return _zero_metrics()
    try:
        with open(predictions_json, "r", encoding="utf-8") as f:
            preds = json.load(f)
    except Exception:
        # If malformed, be conservative and return zero metrics
        return _zero_metrics()
    if not isinstance(preds, list):
        return _zero_metrics()
    if len(preds) == 0:
        return _zero_metrics()

    # Ensure minimal required COCO keys exist in GT dataset
    ds = coco_gt.dataset
    if "info" not in ds:
        ds["info"] = {"description": "auto-filled by evaluate_coco"}
    if "licenses" not in ds:
        ds["licenses"] = []
    for im in ds.get("images", []):
        if "license" not in im:
            im["license"] = 0

    coco_dt = coco_gt.loadRes(preds)  # pass parsed list instead of path
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if img_ids:
        ev.params.imgIds = list(set(img_ids))
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {
        "mAP@0.5:0.95": float(s[0]),
        "AP@0.5": float(s[1]),
        "AP@0.75": float(s[2]),
        "AR_all": float(s[8]),
    }
