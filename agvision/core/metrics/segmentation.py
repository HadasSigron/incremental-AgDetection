from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Iterable, Union
from collections import Counter
from pathlib import Path

import numpy as np
import cv2


# ============================== I/O helpers ===================================

def _load_mask_gray(path: str | Path) -> np.ndarray:
    """
    Load a single-channel mask as int32 labels.
    Assumes masks are stored as PNG/8-bit where pixel value is the class id (0..K-1).
    """
    p = str(path)
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not found or cannot be read: {p}")
    return m.astype(np.int32, copy=False)


def _most_common_shape(shapes: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Return the most frequent (H, W) shape; ties resolved by first occurrence."""
    ctr = Counter(shapes)
    return ctr.most_common(1)[0][0]


def _determine_target_shape(
    gt_paths: List[str],
    pr_paths: List[str],
    params: Optional[Dict[str, Any]],
) -> Tuple[int, int]:
    """
    Decide a common target (H, W) to which all masks will be resized.

    Priority:
      1) params['target_shape'] = [H, W]
      2) params['resize_policy'] in {'max','min','mode'} (default: 'max')
         - 'max'  -> (maxH, maxW)
         - 'min'  -> (minH, minW)
         - 'mode' -> most common shape across GT ∪ Pred
    """
    params = params or {}

    # Explicit target
    if "target_shape" in params and params["target_shape"]:
        H, W = [int(x) for x in params["target_shape"]]
        return (H, W)

    # Scan shapes once
    shapes: List[Tuple[int, int]] = []
    for p in gt_paths:
        m = _load_mask_gray(p)
        shapes.append((m.shape[0], m.shape[1]))
    for p in pr_paths:
        m = _load_mask_gray(p)
        shapes.append((m.shape[0], m.shape[1]))

    if not shapes:
        raise ValueError("No masks found to determine a target shape.")

    policy = str(params.get("resize_policy", "max")).lower()
    if policy == "mode":
        return _most_common_shape(shapes)
    if policy == "min":
        H = min(h for h, _ in shapes)
        W = min(w for _, w in shapes)
        return (H, W)

    # default: 'max'
    H = max(h for h, _ in shapes)
    W = max(w for _, w in shapes)
    return (H, W)


def _resize_mask_nn(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a label mask using nearest-neighbor to preserve class ids."""
    th, tw = target_hw
    if mask.shape[:2] == (th, tw):
        return mask
    return cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)


def _stack_masks(paths: List[str], target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Read masks from paths and resize each to target_hw, then stack to (N, H, W).
    This removes the previous constraint that all inputs must share identical shape.
    """
    arrs: List[np.ndarray] = []
    for p in paths:
        m = _load_mask_gray(p)
        m = _resize_mask_nn(m, target_hw)
        arrs.append(m)
    if not arrs:
        raise ValueError("No masks to stack.")
    return np.stack(arrs, axis=0)  # (N, H, W), dtype=int32


# ============================== metrics core ==================================

def _flatten_valid(
    gt: np.ndarray,
    pr: np.ndarray,
    ignore_index: Optional[Union[int, Iterable[int]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten GT/PRED while dropping ignored labels.
    """
    if gt.shape != pr.shape:
        raise ValueError(f"Shape mismatch gt={gt.shape} vs pr={pr.shape}")
    g = gt.reshape(-1)
    p = pr.reshape(-1)
    if ignore_index is None:
        return g, p
    if isinstance(ignore_index, int):
        mask = (g != ignore_index)
    else:
        ign = set(int(x) for x in ignore_index)
        mask = ~np.isin(g, list(ign))
    return g[mask], p[mask]


def _confusion_matrix(
    gt: np.ndarray,
    pr: np.ndarray,
    num_classes: int,
    ignore_index: Optional[Union[int, Iterable[int]]] = None
) -> np.ndarray:
    """
    Return KxK confusion matrix (rows: GT, cols: Pred), after removing ignored labels.
    """
    g, p = _flatten_valid(gt, pr, ignore_index=ignore_index)
    if g.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    # Filter any out-of-range labels to be robust
    valid = (g >= 0) & (g < num_classes) & (p >= 0) & (p < num_classes)
    g = g[valid]
    p = p[valid]
    idx = g * num_classes + p
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.astype(np.int64, copy=False)


def _safe_div(n: np.ndarray, d: np.ndarray, eps: float) -> np.ndarray:
    """Elementwise safe division with epsilon."""
    return n / (d + eps)


def _per_class_from_cm(cm: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    """
    Compute per-class primitives from a confusion matrix.

    Returns:
      tp, fp, fn, tn, support (all np.ndarray of shape [K])
    """
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)       # GT pixels per class
    pred_count = cm.sum(axis=0).astype(np.float64)    # Predicted pixels per class
    fp = pred_count - tp
    fn = support - tp
    tn = cm.sum() - (tp + fp + fn)
    # Clamp to >= 0 for numerical stability
    fp = np.maximum(fp, 0.0)
    fn = np.maximum(fn, 0.0)
    tn = np.maximum(tn, 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": support}


# ----------- individual metrics (computed from confusion / stacks) ------------

def _pixel_accuracy(gt: np.ndarray, pr: np.ndarray) -> float:
    """
    Pixel Accuracy = correct / total over all pixels.
    Shapes: gt, pr -> (N, H, W) with integer labels.
    """
    if gt.shape != pr.shape:
        raise ValueError(f"Shape mismatch gt={gt.shape} vs pr={pr.shape}")
    correct = (gt == pr).sum(dtype=np.int64)
    total = gt.size
    return float(correct) / float(total) if total else 0.0


def _miou_from_cm(cm: np.ndarray, eps: float = 1e-12) -> float:
    """Mean IoU across classes with non-zero union."""
    stats = _per_class_from_cm(cm, eps=eps)
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    union = tp + fp + fn
    valid = union > 0
    iou = _safe_div(tp, union, eps=eps)
    return float(iou[valid].mean()) if np.any(valid) else 0.0


def _iou_per_class_from_cm(cm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    stats = _per_class_from_cm(cm, eps=eps)
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    union = tp + fp + fn
    return _safe_div(tp, union, eps=eps)


def _mean_accuracy_from_cm(cm: np.ndarray, eps: float = 1e-12) -> float:
    """Mean class accuracy = average(tp / support) over classes with support>0."""
    stats = _per_class_from_cm(cm, eps=eps)
    tp, support = stats["tp"], stats["support"]
    valid = support > 0
    acc = _safe_div(tp, support, eps=eps)
    return float(acc[valid].mean()) if np.any(valid) else 0.0


def _fw_iou_from_cm(cm: np.ndarray, eps: float = 1e-12) -> float:
    """
    Frequency Weighted IoU = sum_i (freq_i * IoU_i) where freq_i = support_i / total.
    """
    total = cm.sum()
    if total <= 0:
        return 0.0
    support = cm.sum(axis=1).astype(np.float64)
    iou_c = _iou_per_class_from_cm(cm, eps=eps)
    freq = support / float(total)
    return float((freq * iou_c).sum())


def _precision_per_class_from_cm(cm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    stats = _per_class_from_cm(cm, eps=eps)
    tp, fp = stats["tp"], stats["fp"]
    return _safe_div(tp, tp + fp, eps=eps)


def _recall_per_class_from_cm(cm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    stats = _per_class_from_cm(cm, eps=eps)
    tp, fn = stats["tp"], stats["fn"]
    return _safe_div(tp, tp + fn, eps=eps)


def _dice_per_class_from_cm(cm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    stats = _per_class_from_cm(cm, eps=eps)
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    return _safe_div(2 * tp, 2 * tp + fp + fn, eps=eps)


# ============================== public API ====================================

SUPPORTED_METRIC_ALIASES = {
    # Global scalars
    "pixel_accuracy": {"pixel_accuracy", "pa"},
    "miou": {"miou", "mean_iou"},
    "mean_accuracy": {"mean_accuracy", "macc", "mpa"},
    "fw_iou": {"fw_iou", "fwiou", "frequency_weighted_iou"},
    "precision": {"precision", "precision_macro"},
    "recall": {"recall", "recall_macro"},
    "dice": {"dice", "f1", "mean_dice"},

    # Per-class expansions (each emits class-wise keys)
    "iou_per_class": {"iou_per_class"},
    "dice_per_class": {"dice_per_class"},
    "precision_per_class": {"precision_per_class"},
    "recall_per_class": {"recall_per_class"},
}

def _normalize_requested(requested: Iterable[str]) -> List[str]:
    """
    Normalize requested metric names to canonical keys defined in SUPPORTED_METRIC_ALIASES.
    Unknown names are ignored to be permissive with user YAMLs.
    """
    out: List[str] = []
    rq = [str(x).strip().lower().replace("-", "_") for x in (requested or [])]
    for canon, aliases in SUPPORTED_METRIC_ALIASES.items():
        if any(r in aliases for r in rq):
            out.append(canon)
    return out


def compute_metrics_from_config(
    y_true_paths: List[str],
    y_pred_paths: List[str],
    requested: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute segmentation metrics given lists of GT/PRED mask file paths.
    Only the metrics listed in `requested` are computed and returned.

    Resizing policy:
      All masks (GT and PRED) are resized to a common shape to support datasets
      where images/masks have different sizes across samples.

    Parameters
    ----------
    y_true_paths : List[str]
        Paths to ground-truth masks (PNG/8-bit: pixel values are class IDs).
    y_pred_paths : List[str]
        Paths to predicted masks.
    requested : List[str]
        List of metric names requested by the caller (from YAML).
        Supported (canonical keys & aliases):
          - pixel_accuracy | pa
          - miou | mean_iou
          - mean_accuracy | macc | mpa
          - fw_iou | fwiou | frequency_weighted_iou
          - precision | precision_macro
          - recall | recall_macro
          - dice | f1 | mean_dice
          - iou_per_class
          - dice_per_class
          - precision_per_class
          - recall_per_class
        Unknown names are ignored (no exception) to keep runs robust.

    params : Dict[str, Any], optional
        Extra controls:
          - target_shape: [H, W]                  # explicit resize target
          - resize_policy: 'max'|'min'|'mode'     # default: 'max'
          - num_classes: int                      # if omitted, inferred as max+1
          - ignore_index: int | List[int]         # optional "void" label(s) to ignore
          - class_names: List[str]                # names for per-class metric keys
          - per_class_prefix: str                 # default: 'class'
          - eps: float                            # numeric stability (default 1e-12)

    Returns
    -------
    Dict[str, float]
        A flat dict of metric_name -> value.
        Per-class metrics are emitted as multiple keys:
          e.g., "iou[class=weed]" or "dice[class=3]" based on `class_names`/indices.
    """
    params = dict(params or {})
    if len(y_true_paths) != len(y_pred_paths):
        raise ValueError(f"GT/PRED length mismatch: {len(y_true_paths)} vs {len(y_pred_paths)}")

    # Decide a common canvas size
    target_hw = _determine_target_shape(y_true_paths, y_pred_paths, params)

    # Read & resize stacks
    gt_stack = _stack_masks(y_true_paths, target_hw)
    pr_stack = _stack_masks(y_pred_paths, target_hw)

    # Prepare meta
    eps: float = float(params.get("eps", 1e-12))
    # If num_classes not provided, infer from max label across GT/PRED
    if "num_classes" in params and params["num_classes"] is not None:
        num_classes = int(params["num_classes"])
    else:
        num_classes = int(max(int(gt_stack.max(initial=0)), int(pr_stack.max(initial=0))) + 1)

    ignore_index = params.get("ignore_index", None)
    class_names: Optional[List[str]] = params.get("class_names")
    per_class_prefix: str = str(params.get("per_class_prefix", "class"))

    # Compute confusion matrix once (over the whole stack)
    cm = _confusion_matrix(gt_stack, pr_stack, num_classes=num_classes, ignore_index=ignore_index)

    # Normalize requested metric names
    req = _normalize_requested(requested)

    out: Dict[str, float] = {}

    # --------- global scalars ----------
    if "pixel_accuracy" in req:
        out["pixel_accuracy"] = _pixel_accuracy(gt_stack, pr_stack)

    if "miou" in req:
        out["miou"] = _miou_from_cm(cm, eps=eps)

    if "mean_accuracy" in req:
        out["mean_accuracy"] = _mean_accuracy_from_cm(cm, eps=eps)

    if "fw_iou" in req:
        out["fw_iou"] = _fw_iou_from_cm(cm, eps=eps)

    if "precision" in req:
        prec_c = _precision_per_class_from_cm(cm, eps=eps)
        # Macro average over classes that have either support>0 or predictions>0
        valid = (cm.sum(axis=1) + cm.sum(axis=0)) > 0
        out["precision"] = float(prec_c[valid].mean()) if np.any(valid) else 0.0

    if "recall" in req:
        rec_c = _recall_per_class_from_cm(cm, eps=eps)
        valid = (cm.sum(axis=1)) > 0
        out["recall"] = float(rec_c[valid].mean()) if np.any(valid) else 0.0

    if "dice" in req:
        dice_c = _dice_per_class_from_cm(cm, eps=eps)
        valid = (cm.sum(axis=1) + cm.sum(axis=0)) > 0
        out["dice"] = float(dice_c[valid].mean()) if np.any(valid) else 0.0

    # --------- per-class expansions ----------
    def _key(metric: str, idx: int) -> str:
        name = (class_names[idx] if class_names and idx < len(class_names) else str(idx))
        return f"{metric}[{per_class_prefix}={name}]"

    if "iou_per_class" in req:
        iou_c = _iou_per_class_from_cm(cm, eps=eps)
        for i, v in enumerate(iou_c.tolist()):
            out[_key("iou", i)] = float(v)

    if "dice_per_class" in req:
        dice_c = _dice_per_class_from_cm(cm, eps=eps)
        for i, v in enumerate(dice_c.tolist()):
            out[_key("dice", i)] = float(v)

    if "precision_per_class" in req:
        prec_c = _precision_per_class_from_cm(cm, eps=eps)
        for i, v in enumerate(prec_c.tolist()):
            out[_key("precision", i)] = float(v)

    if "recall_per_class" in req:
        rec_c = _recall_per_class_from_cm(cm, eps=eps)
        for i, v in enumerate(rec_c.tolist()):
            out[_key("recall", i)] = float(v)

    return out
