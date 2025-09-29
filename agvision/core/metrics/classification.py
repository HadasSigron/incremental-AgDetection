"""Classification metrics (placeholder)."""


from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

# Add to: agvision/core/metrics/classification.py
from typing import Dict, Iterable, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
)

def _topk_accuracy_from_probs(y_true: List[int], probs: List[List[float]], k: int) -> float:
    """
    Compute top-k accuracy given per-class probabilities/scores.
    Assumes probs shape = [N, C] and class ids in [0..C-1].
    """
    arr = np.asarray(probs, dtype=float)              # (N, C)
    true = np.asarray(y_true, dtype=int)              # (N,)
    if arr.ndim != 2 or arr.shape[0] != true.shape[0]:
        raise ValueError("probs shape must be [N, C] aligned with y_true length.")
    if k < 1 or k > arr.shape[1]:
        raise ValueError(f"Invalid top-k={k} for C={arr.shape[1]}.")
    # argpartition is O(N*C) and faster than full argsort; returns unordered top-k indices
    topk = np.argpartition(arr, kth=-k, axis=1)[:, -k:]  # (N, k) class indices
    correct = np.any(topk == true[:, None], axis=1)      # (N,)
    return float(np.mean(correct))


def compute_metrics_from_config(
    y_true: List[int],
    y_pred: List[int],
    *,
    probs: Optional[List[List[float]]] = None,
    requested: Iterable[str],
    average: str = "macro",
    strict: bool = True,
) -> Dict[str, float]:
    """
    Compute exactly the metrics requested by config.

    Supported names (case-insensitive):
      - "accuracy", "precision", "recall", "f1"
      - "balanced_accuracy", "mcc"
      - "top1", "top2", ..., "topK"      (requires probs)
      - "auroc_ovr", "auroc_ovo"         (requires probs; multi-class)

    Behavior:
      * If 'strict' is True, unknown names or missing prerequisites (e.g., no 'probs'
        for 'top5') raise KeyError/ValueError.
      * If 'strict' is False, such metrics are silently skipped.

    Notes:
      * 'average' parameter applies to precision/recall/f1 and AUROC averaging.
    """
    # Base set (label-only)
    available: Dict[str, float] = {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "precision":          float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall":             float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1":                 float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":                float(matthews_corrcoef(y_true, y_pred)),
    }

    req_lower = [str(m).lower() for m in requested]

    # Top-K (requires probs)
    if any(name.startswith("top") and name[3:].isdigit() for name in req_lower):
        if probs is None:
            if strict:
                raise KeyError("Requested top-k metrics but 'probs' are not available from the model.")
        else:
            # compute only the requested K values
            ks = sorted({int(name[3:]) for name in req_lower if name.startswith("top") and name[3:].isdigit()})
            for k in ks:
                available[f"top{k}"] = _topk_accuracy_from_probs(y_true, probs, k)

    # AUROC (requires probs)
    if ("auroc_ovr" in req_lower or "auroc_ovo" in req_lower) and probs is not None:
        y_true_arr = np.asarray(y_true, dtype=int)
        scores = np.asarray(probs, dtype=float)  # (N, C)
        # Binarize using all classes in scores: assumes labels are in [0..C-1]
        C = scores.shape[1]
        if C < 2:
            if strict:
                raise ValueError("AUROC requires at least 2 classes.")
        else:
            # one-vs-rest / one-vs-one multi-class AUROC
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true_arr, classes=np.arange(C))
            if "auroc_ovr" in req_lower:
                available["auroc_ovr"] = float(roc_auc_score(y_bin, scores, average=average, multi_class="ovr"))
            if "auroc_ovo" in req_lower:
                available["auroc_ovo"] = float(roc_auc_score(y_bin, scores, average=average, multi_class="ovo"))
    elif ("auroc_ovr" in req_lower or "auroc_ovo" in req_lower) and probs is None and strict:
        raise KeyError("Requested AUROC metrics but 'probs' are not available from the model.")

    # Filter strictly to requested
    result: Dict[str, float] = {}
    unknown: List[str] = []
    for name in req_lower:
        if name in available:
            result[name] = available[name]
        elif name.startswith("top") and name[3:].isdigit():
            # Computed above if probs existed; if missing here, it's an error in strict mode
            if strict:
                raise KeyError(f"Requested metric '{name}' could not be computed.")
        else:
            unknown.append(name)

    if unknown and strict:
        raise KeyError(f"Unknown metric names requested: {unknown}. "
                       f"Supported: {sorted(list(set(list(available.keys()) + ['topK', 'auroc_ovr', 'auroc_ovo'])))}")

    return result




def _per_class_counts(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TP/FP/FN per class for multi-class classification.

    Args:
        y_true: Ground-truth labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        classes: Unique class ids, shape (C,)

    Returns:
        tp: True positives per class, shape (C,)
        fp: False positives per class, shape (C,)
        fn: False negatives per class, shape (C,)
    """
    C = classes.shape[0]
    tp = np.zeros(C, dtype=np.int64)
    fp = np.zeros(C, dtype=np.int64)
    fn = np.zeros(C, dtype=np.int64)

    for i, c in enumerate(classes):
        tp[i] = int(((y_pred == c) & (y_true == c)).sum())
        fp[i] = int(((y_pred == c) & (y_true != c)).sum())
        fn[i] = int(((y_pred != c) & (y_true == c)).sum())
    return tp, fp, fn


def _safe_div(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """
    Element-wise safe division with zero-division handling.
    """
    out = np.full_like(a, fill, dtype=np.float64)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out


def compute_prf1_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute Accuracy, Precision, Recall, and F1 for multi-class classification
    without external dependencies (sklearn-agnostic).

    Args:
        y_true: Ground-truth labels
        y_pred: Predicted labels
        average: One of {"macro", "micro", "weighted"}.

    Returns:
        {
          "accuracy": float,
          "precision": float,
          "recall": float,
          "f1": float
        }
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    if yt.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    classes = np.unique(np.concatenate([yt, yp], axis=0))
    C = classes.shape[0]
    acc = float((yt == yp).mean())

    # Per-class counts
    tp, fp, fn = _per_class_counts(yt, yp, classes)
    precision_c = _safe_div(tp, tp + fp, fill=0.0)
    recall_c    = _safe_div(tp, tp + fn, fill=0.0)
    f1_c_num    = 2.0 * precision_c * recall_c
    f1_c_den    = precision_c + recall_c
    f1_c        = _safe_div(f1_c_num, f1_c_den, fill=0.0)

    supports = np.array([(yt == c).sum() for c in classes], dtype=np.int64)
    total_support = int(supports.sum()) if supports.size > 0 else 0

    if average == "micro":
        # Micro-averaging: aggregate counts first, then compute metrics
        TP = int(tp.sum())
        FP = int(fp.sum())
        FN = int(fn.sum())
        prec = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        rec  = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        f1   = float(0.0 if (prec + rec) == 0.0 else (2 * prec * rec) / (prec + rec))
    elif average == "weighted" and total_support > 0:
        w = supports / total_support
        prec = float((w * precision_c).sum())
        rec  = float((w * recall_c).sum())
        f1   = float((w * f1_c).sum())
    else:
        # default macro (simple mean)
        prec = float(precision_c.mean()) if C > 0 else 0.0
        rec  = float(recall_c.mean())    if C > 0 else 0.0
        f1   = float(f1_c.mean())        if C > 0 else 0.0

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
