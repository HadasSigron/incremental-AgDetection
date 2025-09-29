from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import traceback

# Qt (prefer PyQt6; allow PyQt5)
try:  # pragma: no cover
    from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
except Exception:  # pragma: no cover
    from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from core.datasets.factory import build_dataset_adapter
from core.runners.registry import get_runner
from ui.workers.eval_worker_merge import build_appconfig


# ----------------------------- helpers -----------------------------

def _normalize_result(result: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize Runner result to (metrics, artifacts)."""
    if isinstance(result, tuple) and len(result) == 2:
        return result[0] or {}, result[1] or {}
    if isinstance(result, dict):
        return result.get("metrics", {}) or {}, result.get("artifacts", {}) or {}
    metrics = getattr(result, "metrics", {}) or {}
    artifacts = getattr(result, "artifacts", {}) or {}
    return metrics, artifacts


def _safe_get(obj: Any, *path: str, default: Any = None) -> Any:
    """Nested getattr/get lookup with a default."""
    cur = obj
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, default if p == path[-1] else None)
        else:
            cur = getattr(cur, p, default if p == path[-1] else None)
    return cur if cur is not None else default


def _model_label(cfg: Any) -> str:
    """
    Produce a concise model label for the table.
    Tries: model.name → weights filename stem → '?'.
    """
    name = _safe_get(cfg, "model", "name", default=None)
    if name:
        return str(name)
    w = _safe_get(cfg, "model", "weights", default=None)
    try:
        return Path(str(w)).stem if w else "?"
    except Exception:
        return "?"


# ----------------------------- API -----------------------------

class WorkerSignals(QObject):
    """
    Qt signals for long-running evaluation jobs.

    started  : ()
    progress : (int,)
    message  : (str,)
    error    : (str,)
    result   : (object,)   # single dict payload
    finished : (object,)   # final payload (same shape)
    """
    started = pyqtSignal()
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    finished = pyqtSignal(object)


class EvalWorker(QRunnable):
    """
    QRunnable that executes a model evaluation on a background thread.
    Emits a SINGLE dict payload shaped for the UI (MetricsTable).
    """

    def __init__(
        self,
        # YAML-driven inputs:
        benchmark_yaml: Optional[str] = None,
        auto_model: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,

        # Pre-built path (not used here):
        task: Optional[str] = None,
        dataset: Any | None = None,
        config: Dict[str, Any] | Any | None = None,

        # Cancellation:
        cancel_token: Any | None = None,
    ) -> None:
        super().__init__()
        self._mode = "yaml" if benchmark_yaml else "prebuilt"
        self.benchmark_yaml = benchmark_yaml
        self.auto_model = auto_model or {}
        self.model_name = model_name
        self.overrides = overrides or {}

        self.task = task
        self.dataset = dataset
        self.config = config

        self.cancel_token = cancel_token
        self.signals = WorkerSignals()

    def run(self) -> None:
        self.signals.started.emit()
        self.signals.message.emit("[EvalWorker] Starting evaluation...")
        try:
            if self._mode != "yaml":
                raise ValueError("EvalWorker: YAML mode is required.")

            # 1) Build config from YAML (+auto-model/+overrides)
            cfg = build_appconfig(
                benchmark_yaml=str(self.benchmark_yaml),
                auto_model=(self.auto_model or None),
                model_name=(self.model_name or None),
                overrides=self.overrides,
            )

            # Log a concise summary
            try:
                self.signals.message.emit(
                    "[EvalWorker] Config → "
                    f"task={getattr(cfg, 'task', '?')}, "
                    f"benchmark={getattr(cfg, 'benchmark_name', '?')}, "
                    f"dataset.kind={_safe_get(cfg, 'dataset', 'kind', default='?')}, "
                    f"img_dir={_safe_get(cfg, 'dataset', 'img_dir', default='?')}, "
                    f"mask_dir/ann={_safe_get(cfg, 'dataset', 'mask_dir', default=_safe_get(cfg, 'dataset', 'ann_file', default='?'))}, "
                    f"metrics={list(_safe_get(cfg, 'eval', 'metrics', default=[]))}"
                )
            except Exception:
                pass

            # 2) Dataset

            dataset = build_dataset_adapter(cfg)

            # לוג כללי מתוך האדאפטר (describe)
            desc = getattr(dataset, "describe", lambda: {})()
            if desc:
                self.signals.message.emit("[EvalWorker] Resolved paths:")
                for k, v in desc.items():
                    self.signals.message.emit(f"  {k}={v}")

            # בדיקת גודל
            try:
                n = len(dataset)
            except Exception:
                n = -1
            self.signals.message.emit(f"[EvalWorker] Dataset size: {n} samples.")
            if n <= 0:
                raise RuntimeError("Dataset is empty. Check dataset paths / matching file names / COCO instances JSON.")

            # 3) Runner
            runner = get_runner(cfg.task)
            self.signals.message.emit(f"[EvalWorker] Using runner: {runner.__class__.__name__}")
            raw = runner.run(dataset, cfg, getattr(self, "cancel_token", None))
            metrics, artifacts = _normalize_result(raw)

            # 4) Duration into metrics (for convenience)
            if "duration" not in metrics and isinstance(artifacts, dict) and artifacts.get("runtime_sec") is not None:
                metrics = dict(metrics)
                try:
                    metrics["duration"] = float(artifacts["runtime_sec"])
                except Exception:
                    metrics["duration"] = artifacts["runtime_sec"]

            # 5) Persist run_config.json (best-effort)
            out_dir = (artifacts or {}).get("run_dir") or (artifacts or {}).get("out_dir")
            if out_dir:
                try:
                    (Path(out_dir) / "run_config.json").write_text(
                        cfg.model_dump_json(indent=2) if hasattr(cfg, "model_dump_json") else str(cfg),
                        encoding="utf-8"
                    )
                    self.signals.message.emit(f"[EvalWorker] Saved run_config.json to: {Path(out_dir) / 'run_config.json'}")
                except Exception:
                    pass

            # 6) Compose a single payload the UI expects
            requested_metrics = list(_safe_get(cfg, "eval", "metrics", default=[]))
            payload = {
                "task": str(getattr(getattr(cfg, "task", None), "value", getattr(cfg, "task", "Unknown"))),
                "benchmark": str(getattr(cfg, "benchmark_name", "")),
                "requested_metrics": requested_metrics,          # RAW keys from YAML
                "metrics": metrics,                              # raw metrics dict from runner
                "artifacts": artifacts or {},
                "meta": {
                    "model": _model_label(cfg),
                    "format": str(_safe_get(cfg, "model", "format", default="?")).lower(),
                    "arch": str(_safe_get(cfg, "model", "name", default="")) or str(_safe_get(cfg, "model", "arch", default="?")),
                    "duration": metrics.get("duration") or (artifacts or {}).get("runtime_sec"),
                },
            }

            # 7) Emit
            self.signals.result.emit(payload)
            self.signals.finished.emit(payload)

        except Exception:
            tb = traceback.format_exc()
            self.signals.error.emit(tb)
            self.signals.finished.emit({"cancelled": True, "error": tb})
