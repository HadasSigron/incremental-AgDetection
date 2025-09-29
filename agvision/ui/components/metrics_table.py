from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable, Literal

# Prefer PyQt6; allow PyQt5 fallback
try:  # pragma: no cover
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
    )
except Exception:  # pragma: no cover
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
    )

from ui.state.app_state import AppState

FilterMode = Literal["current", "all"]


# ----------------------------- helpers -----------------------------

def _display_name(key: str) -> str:
    """
    Map a raw YAML metric key to a friendly column header.
    """
    k = (key or "").strip()
    low = k.lower()
    mapping = {
        "miou": "mIoU",
        "mean_iou": "mIoU",
        "pixel_accuracy": "Pixel Accuracy",
        "pa": "Pixel Accuracy",
        "dice": "Dice",
        "f1": "F1",
        "f1_score": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "ap50": "AP@0.5",
        "ap5095": "AP@0.5:0.95",
        "map@0.5": "mAP@0.5",
        "map@0.5:0.95": "mAP@0.5:0.95",
        "map50": "mAP@0.5",
        "map5095": "mAP@0.5:0.95",
        "runtime_sec": "Run time (s)",
        "duration": "Run time (s)",
    }
    if low in mapping:
        return mapping[low]
    if low.startswith("map@") or low.startswith("map"):
        return "m" + k[1:] if k and k[0] in ("m", "M") else f"m{k}"
    return k.replace("_", " ").title()


def _unique(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            out.append(s); seen.add(s)
    return out


def _is_number_like(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


# canonical (lower) → alternate keys (lower) for value resolution
_ALIASES: Dict[str, List[str]] = {
    "miou": ["miou", "mean_iou"],
    "pixel_accuracy": ["pixel_accuracy", "pa", "pixel accuracy"],
    "map@0.5": ["map@0.5", "map50", "ap50", "ap@0.5"],
    "map@0.5:0.95": ["map@0.5:0.95", "map5095", "ap5095", "ap@0.5:0.95"],
    "runtime_sec": ["runtime_sec", "duration", "run time (s)"],
    "duration": ["duration", "runtime_sec", "run time (s)"],
    "f1": ["f1", "f1_score"],
}

def _lookup_metric(raw_key: str, metrics: Dict[str, Any], metrics_display: Optional[Dict[str, Any]] = None) -> Any:
    """
    Resolve a value for a YAML *raw* key from:
      1) metrics (exact / case-insensitive / aliases)
      2) metrics_display (matched by display title)
    """
    metrics = metrics or {}
    ml = {str(k).lower(): v for k, v in metrics.items()}

    if raw_key in metrics:
        return metrics[raw_key]
    v = ml.get(raw_key.lower())
    if v is not None:
        return v
    for alt in _ALIASES.get(raw_key.lower(), []):
        if alt in ml:
            return ml[alt]
    if metrics_display:
        disp = _display_name(raw_key)
        if disp in metrics_display:
            return metrics_display[disp]
        v2 = {str(k).lower(): v for k, v in metrics_display.items()}.get(disp.lower())
        if v2 is not None:
            return v2
    return None


def _coerce_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(y) for y in x]
    return [t.strip() for t in str(x).split(",") if t.strip()]


# ============================== Widget ==============================

class MetricsTable(QWidget):
    """
    Results table that NEVER hard-codes metric columns.

    • Metric columns are always taken from the selected YAML's `eval.metrics`.
    • If YAML keys are not available (rare), columns are derived from the first
      incoming metrics dictionary (fallback), and future runs can expand them.
    • Accepts result payloads in multiple signatures:
        - append_result(payload_dict)
        - append_result(task, benchmark, metrics[, meta])
        - append_result(meta_dict, metrics_display_dict, [benchmark_id])  <-- your screen does this
        - append_result(**kwargs)
    """

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state

        # Append-only storage
        self._all: List[Dict[str, Any]] = []

        # Filtering
        self._filter_mode: FilterMode = "all"   # don't hide rows when no id yet
        self._current_benchmark_id: Optional[str] = None

        # Column schema
        self._meta_cols: List[str] = ["Model", "Format", "Arch", "Run time"]
        self._metric_raw_keys: List[str] = self._read_metric_keys_from_state()  # raw YAML keys
        self._metric_cols: List[str] = [_display_name(k) for k in self._metric_raw_keys]
        self.columns: List[str] = self._meta_cols + self._metric_cols

        # UI
        self._title = QLabel("Results")
        self.table = QTableWidget(0, len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        header = self.table.horizontalHeader()
        try:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # PyQt6
        except Exception:  # pragma: no cover
            header.setSectionResizeMode(QHeaderView.Stretch)             # PyQt5

        layout = QVBoxLayout(self)
        layout.addWidget(self._title)
        layout.addWidget(self.table)

    # ---------------------------- Public API -----------------------------

    def set_task_type(self, _task: Optional[str]) -> None:
        """Kept for backward compatibility; not used here."""
        return

    def set_current_benchmark(self, benchmark_id: Optional[str]) -> None:
        self._current_benchmark_id = (str(benchmark_id) if benchmark_id else None)
        self._refresh_view()

    def set_filter_mode(self, mode: FilterMode) -> None:
        if mode not in ("current", "all"):
            return
        self._filter_mode = mode
        self._refresh_view()

    def set_metric_keys(self, raw_metric_keys: List[str]) -> None:
        """Rebuild metric columns using RAW keys from YAML."""
        raw = _unique([str(k).strip() for k in (raw_metric_keys or []) if str(k).strip()])
        if raw == self._metric_raw_keys:
            return
        self._metric_raw_keys = raw
        self._metric_cols = [_display_name(k) for k in self._metric_raw_keys]
        self.columns = self._meta_cols + self._metric_cols
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        self._refresh_view()

    def refresh_metric_keys_from_state(self) -> None:
        """Pull RAW YAML keys from AppState and rebuild columns."""
        self.set_metric_keys(self._read_metric_keys_from_state())

    def clear_all(self) -> None:
        self._all.clear()
        self.table.setRowCount(0)

    # ---------------------- Flexible result intake -----------------------

    def append_result(self, *args: Any, **kwargs: Any) -> None:
        """
        Robust slot handling all common signatures (see class docstring).
        """
        # Always prefer YAML at the moment of appending (in case user switched YAML)
        yaml_keys = self._read_metric_keys_from_state()
        if yaml_keys:
            self.set_metric_keys(yaml_keys)

        payload: Dict[str, Any] = {}

        # Case 1: payload dict
        if len(args) == 1 and isinstance(args[0], dict):
            payload.update(args[0])

        # Case 2: (task, benchmark, metrics[, meta])
        elif len(args) >= 2 and all(isinstance(a, (str, type(None))) for a in args[:2]):
            payload["task"] = args[0] or None
            payload["benchmark"] = args[1] or None
            if len(args) >= 3 and isinstance(args[2], dict):
                payload["metrics"] = args[2]
            if len(args) >= 4 and isinstance(args[3], dict):
                payload["meta"] = args[3]

        # Case 3: (meta_dict, metrics_display_dict, [benchmark_id])  <-- your screen
        elif len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            # Heuristic: treat the first dict as meta if it *looks* like meta
            meta_keys = {"model", "format", "arch", "architecture", "duration"}
            if meta_keys.intersection({k.lower() for k in map(str, args[0].keys())}):
                payload["meta"] = args[0]
                payload["metrics_display"] = args[1]
                if len(args) >= 3 and (isinstance(args[2], str) or args[2] is None):
                    payload["benchmark"] = args[2]
            else:
                # If not meta-like, fall back to treating them as metrics + meta
                payload["metrics"] = args[0]
                payload["meta"] = args[1]

        # Merge kwargs (highest precedence)
        payload.update(kwargs or {})

        # --- Resolve columns if worker provided requested_metrics explicitly ---
        req = _coerce_list(payload.get("requested_metrics"))
        if req:
            self.set_metric_keys(req)

        # --- Extract pieces ---
        meta_in = payload.get("meta", {}) or {}
        artifacts = payload.get("artifacts", {}) or {}
        metrics_in: Dict[str, Any] = payload.get("metrics", {}) or {}
        metrics_display = payload.get("metrics_display", {}) or {}

        # If we still have no YAML columns, fallback to first incoming metrics keys
        if not self._metric_raw_keys:
            inferred = list(metrics_in.keys()) or list(metrics_display.keys())
            self.set_metric_keys(inferred)

        # Union any newly-seen raw metric keys so nothing gets dropped
        unseen = [k for k in metrics_in.keys() if str(k).strip() and str(k).strip() not in self._metric_raw_keys]
        if unseen:
            self.set_metric_keys(self._metric_raw_keys + unseen)

        # Meta (left columns)
        duration = meta_in.get("duration", artifacts.get("runtime_sec"))
        meta = {
            "model": meta_in.get("model") or "?",
            "format": (str(meta_in.get("format") or "").lower() or "?"),
            "arch": meta_in.get("arch") or meta_in.get("architecture") or "?",
            "duration": f"{float(duration):.3f}s" if _is_number_like(duration) else (str(duration) if duration is not None else "-"),
        }

        # Build a display dict strictly in YAML/display order
        row_metrics: Dict[str, Any] = {}
        for raw_key in self._metric_raw_keys:
            row_metrics[_display_name(raw_key)] = _lookup_metric(raw_key, metrics_in, metrics_display)

        benchmark_id = payload.get("benchmark") or payload.get("benchmark_id") or self._current_benchmark_id

        self._all.append({
            "benchmark": str(benchmark_id) if benchmark_id else None,
            "meta": meta,
            "metrics": row_metrics,
        })
        self._refresh_view()

    # ----------------------------- internals ------------------------------

    def _read_metric_keys_from_state(self) -> List[str]:
        """
        Extract RAW YAML keys from AppState at runtime.
        Tries:
          1) state.current_config.eval.metrics / state.config.eval.metrics
          2) state.eval.metrics
          3) state.selected_benchmark['eval']['metrics']
        """
        s = self.state

        try:
            cfg = getattr(s, "current_config", None) or getattr(s, "config", None)
            if cfg is not None:
                eval_obj = getattr(cfg, "eval", None) or (cfg.get("eval") if isinstance(cfg, dict) else None)
                if eval_obj is not None:
                    metrics = getattr(eval_obj, "metrics", None) or (eval_obj.get("metrics") if isinstance(eval_obj, dict) else None)
                    m = _coerce_list(metrics)
                    if m:
                        return _unique(m)
        except Exception:
            pass

        try:
            eval_obj = getattr(s, "eval", None)
            if eval_obj is not None:
                metrics = getattr(eval_obj, "metrics", None) or (eval_obj.get("metrics") if isinstance(eval_obj, dict) else None)
                m = _coerce_list(metrics)
                if m:
                    return _unique(m)
        except Exception:
            pass

        try:
            sel = getattr(s, "selected_benchmark", None)
            if isinstance(sel, dict):
                m = _coerce_list(sel.get("eval", {}).get("metrics"))
                if m:
                    return _unique(m)
        except Exception:
            pass

        return []

    def _filtered_rows(self) -> List[Dict[str, Any]]:
        if self._filter_mode == "all":
            return list(self._all)
        if not self._current_benchmark_id:
            return list(self._all)
        return [row for row in self._all if row.get("benchmark") == self._current_benchmark_id]

    def _refresh_view(self) -> None:
        rows = self._filtered_rows()
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)

        for r, row in enumerate(rows):
            meta = row["meta"]
            mdisp = row["metrics"]

            # Meta cells
            left = [meta.get("model", "?"), meta.get("format", "?"), meta.get("arch", "?"), meta.get("duration", "-")]
            for c, val in enumerate(left):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))

            # Metric cells (exact YAML/display order)
            for c, disp in enumerate(self._metric_cols, start=len(self._meta_cols)):
                v = mdisp.get(disp)
                if v is None:
                    s = ""
                elif isinstance(v, float):
                    s = f"{v:.4f}"
                elif _is_number_like(v):
                    s = str(v)
                else:
                    s = str(v)
                self.table.setItem(r, c, QTableWidgetItem(s))
