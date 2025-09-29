# core/runners/classifier_runner.py
from __future__ import annotations

import os
import json
import time
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.runners.base import BaseRunner
from core.config.schema import AppConfig, ModelFormat
from core.models.registry import load_classifier
from core.types.runner_result import RunnerResult, ensure_runner_result
from core.metrics.classification import compute_metrics_from_config

# ------------------------------- helpers --------------------------------------

def _is_cancelled(token: Any | None) -> bool:
    """Best-effort cancellation check. Supports tokens with .is_set() or truthy flags."""
    try:
        return bool(token and token.is_set())
    except Exception:
        return bool(token)


def _iter_samples(dataset: Any) -> Iterable[Dict[str, Any]]:
    """
    Normalized access over dataset adapters.

    Expected per-sample keys (classification):
      { "id": int, "path": str, "label": int }
    """
    if hasattr(dataset, "iter_samples"):
        return dataset.iter_samples()
    return iter(dataset)


def _safe_str(x: object) -> str:
    """
    Convert enums / objects to a clean string for path segments.
    - If x has `.value`, use it; else use str(x).
    - Strip/replace path-unfriendly characters.
    """
    try:
        val = getattr(x, "value", x)
    except Exception:
        val = x
    s = str(val)
    return s.replace(os.sep, "_").replace(":", "_").strip()


def _task_name(_: AppConfig) -> str:
    return "Classification"


def _resolve_domain(cfg: AppConfig) -> str:
    if hasattr(cfg, "domain") and cfg.domain is not None:
        return _safe_str(cfg.domain)
    if hasattr(cfg, "benchmark") and getattr(cfg.benchmark, "domain", None) is not None:
        return _safe_str(cfg.benchmark.domain)
    return "default"


def _resolve_benchmark(cfg: AppConfig) -> str:
    for key in ("benchmark_id", "benchmark_name"):
        if hasattr(cfg, key) and getattr(cfg, key):
            return _safe_str(getattr(cfg, key))
    if hasattr(cfg, "benchmark"):
        for key in ("id", "name"):
            if hasattr(cfg.benchmark, key) and getattr(cfg.benchmark, key):
                return _safe_str(getattr(cfg.benchmark, key))
    return "benchmark"


def _runs_root(cfg: AppConfig) -> Path:
    try:
        root = getattr(getattr(cfg, "outputs", None), "root_dir", None)
        if root:
            return Path(str(root))
    except Exception:
        pass
    try:
        root = getattr(getattr(cfg, "paths", None), "runs_root", None)
        if root:
            return Path(str(root))
    except Exception:
        pass
    return Path("runs")


def _out_dir(cfg: AppConfig) -> Path:
    """
    Create: runs/<Task>/<Domain>/<Benchmark>/<YYYYmmdd_HHMMSS>/
    If cfg.eval.params['out_dir'] is provided, use it as-is.
    """
    params = getattr(cfg.eval, "params", {}) if hasattr(cfg, "eval") else {}
    user_out = params.get("out_dir")
    if user_out:
        p = Path(str(user_out))
        p.mkdir(parents=True, exist_ok=True)
        return p

    root = _runs_root(cfg)
    out = root / _task_name(cfg) / _resolve_domain(cfg) / _resolve_benchmark(cfg) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _dynamic_import(script_path: Path) -> Any:
    """
    Import a Python module from a filesystem path (no package install required).

    The module is expected to expose a callable entrypoint (default: 'build')
    that returns a classifier object implementing either:
      - predict_batch(paths: List[str]) -> tuple[List[int], Optional[List[List[float]]]]
      - predict(path: str) -> int
    """
    sp = script_path.resolve()
    if not sp.exists():
        raise FileNotFoundError(f"Script plugin not found: {sp}")
    spec = importlib.util.spec_from_file_location(sp.stem, sp)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for: {sp}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _build_classifier_from_script(cfg: AppConfig) -> Any:
    """
    Build a classifier from a user script.

    Lookup order:
      - cfg.model.extra['script_path']
      - models/<cfg.model.name>.py

    Entrypoint name:
      - cfg.model.extra['entrypoint'] or 'build'
    """
    extra = dict(cfg.model.extra or {})
    entry_name = str(extra.get("entrypoint", "build"))
    script_path = Path(str(extra["script_path"])) if extra.get("script_path") else (Path("models") / f"{cfg.model.name}.py")

    module = _dynamic_import(script_path)
    if not hasattr(module, entry_name):
        raise AttributeError(
            f"Entrypoint '{entry_name}' not found in {script_path}. "
            f"Expected a callable that returns a model."
        )
    return getattr(module, entry_name)(cfg)


def _load_classifier(cfg: AppConfig) -> Any:
    """
    Build/Load a classifier according to cfg.model.format:

      - script      → dynamic plugin file (models/<name>.py)
      - onnx/timm/… → via core.models.registry.load_classifier(opts)
    """
    fmt = cfg.model.format

    if fmt == ModelFormat.script:
        return _build_classifier_from_script(cfg)

    # Registry-backed loaders (e.g., '.onnx' -> core.models.onnx_classifier.OnnxClassifier)
    weights = (getattr(cfg.model, "weights", None) or cfg.model.path)
    weights = Path(str(weights)).as_posix()

    device = (getattr(cfg.model, "device", None) or getattr(cfg.eval, "device", None) or "cpu")
    imgsz = int(getattr(cfg.model, "imgsz", getattr(cfg.dataset, "input_size", 224)) or 224)

    # Optional normalization and behavior (model.* overrides eval.params.*)
    params = dict(getattr(cfg.eval, "params", {}) or {})
    extra  = dict(getattr(cfg.model, "extra", {}) or {})
    mean = tuple(extra.get("mean", params.get("mean", (0.485, 0.456, 0.406))))
    std  = tuple(extra.get("std",  params.get("std",  (0.229, 0.224, 0.225))))
    return_probs = bool(extra.get("return_probs", params.get("return_probs", True)))

    return load_classifier({
        "model_path": weights,
        "device": device,
        "imgsz": imgsz,
        "mean": mean,
        "std":  std,
        "return_probs": return_probs,
    })


# --------------------------------- runner --------------------------------------

class ClassifierRunner(BaseRunner):
    """
    AppConfig-native classification runner.

    Modes:
      (A) Eval-only:
          If cfg.eval.params['predictions_json'] is provided, load predictions and
          compute metrics without running inference.

      (B) Inference:
          Otherwise, build a model by cfg.model.format and run batched inference.

    Outputs:
      - metrics: accuracy / precision / recall / f1 (see compute_metrics_from_config)
      - artifacts:
          * predictions_json        (per-sample predictions [+ optional probs])
          * runtime_sec             (wall-clock runtime as a string)
          * run_dir                 (output directory created for this run)
    """

    def run(self, dataset: Any, cfg: AppConfig, cancel_token: Any | None = None) -> RunnerResult:
        t0 = time.time()
        out_dir = _out_dir(cfg)
        preds_path = out_dir / "predictions.json"

        # --------- (A) Eval-only: use precomputed predictions ----------
        pred_json_path = getattr(cfg.eval, "params", {}).get("predictions_json")
        if pred_json_path:
            preds = json.loads(Path(str(pred_json_path)).read_text(encoding="utf-8"))
            y_true = [int(p["label"]) for p in preds]
            y_pred = [int(p["pred"]) for p in preds]

            probs_list = [p.get("probs") for p in preds]
            probs = None
            if probs_list and all(isinstance(pr, list) and len(pr) > 0 for pr in probs_list):
                probs = probs_list  # type: ignore[assignment]

            avg = str(getattr(getattr(cfg, "eval", None), "params", {}).get("average", "macro"))
            requested = list(getattr(getattr(cfg, "eval", None), "metrics", []) or ["accuracy", "precision", "recall", "f1"])
            strict = bool(getattr(getattr(cfg, "eval", None), "params", {}).get("metrics_strict", True))

            metrics = compute_metrics_from_config(
                y_true, y_pred, probs=probs, requested=requested, average=avg, strict=strict
            )

            return ensure_runner_result({
                "metrics": metrics,
                "artifacts": {
                    "predictions_json": str(pred_json_path),
                    "runtime_sec": f"{time.time() - t0:.3f}",
                    "run_dir": str(out_dir),
                },
            })

        # --------- (B) Inference ----------
        model = _load_classifier(cfg)
        # UI batch_size groups file I/O; the backend may internally chunk to static batch if needed.
        bs = int(getattr(getattr(cfg, "eval", None), "batch_size", 16) or 16)

        y_true: List[int] = []
        y_pred: List[int] = []
        all_preds: List[Dict[str, Any]] = []

        batch: List[Dict[str, Any]] = []
        for sample in _iter_samples(dataset):
            if _is_cancelled(cancel_token):
                break
            batch.append(sample)
            if len(batch) >= bs:
                self._flush_batch(model, batch, y_true, y_pred, all_preds)
                batch.clear()

        if batch and not _is_cancelled(cancel_token):
            self._flush_batch(model, batch, y_true, y_pred, all_preds)

        preds_path.write_text(json.dumps(all_preds, ensure_ascii=False), encoding="utf-8")

        # Compute metrics
        avg = str(getattr(getattr(cfg, "eval", None), "params", {}).get("average", "macro"))
        requested = list(getattr(getattr(cfg, "eval", None), "metrics", []) or ["accuracy", "precision", "recall", "f1"])

        probs_list = [p.get("probs") for p in all_preds]
        probs = None
        if probs_list and all(isinstance(pr, list) and len(pr) > 0 for pr in probs_list):
            probs = probs_list  # type: ignore[assignment]

        strict = bool(getattr(getattr(cfg, "eval", None), "params", {}).get("metrics_strict", True))
        metrics = compute_metrics_from_config(
            y_true, y_pred, probs=probs, requested=requested, average=avg, strict=strict
        )

        return ensure_runner_result({
            "metrics": metrics,
            "artifacts": {
                "predictions_json": str(preds_path),
                "runtime_sec": f"{time.time() - t0:.3f}",
                "run_dir": str(out_dir),
            },
        })

    # ---------------------------- internal ---------------------------------

    @staticmethod
    def _flush_batch(model: Any, batch: List[Dict[str, Any]],
                     y_true: List[int], y_pred: List[int],
                     out_json: List[Dict[str, Any]]) -> None:
        """
        Run a batch through the model and append results to collectors.

        Model contract:
          - Preferred: predict_batch(paths) -> (preds, probs?)  # probs is optional
          - Fallback:  predict(path) -> int                     # single-file API
        """
        paths = [str(s["path"]) for s in batch]
        labels = [int(s["label"]) for s in batch]

        preds: List[int]
        probs: Optional[List[List[float]]] = None

        if hasattr(model, "predict_batch"):
            out = model.predict_batch(paths)
            if isinstance(out, tuple) and len(out) == 2:
                preds, probs = out  # type: ignore[assignment]
            else:
                preds = list(out)  # type: ignore[assignment]
        else:
            preds = [int(model.predict(p)) for p in paths]  # type: ignore[attr-defined]

        y_true.extend(labels)
        y_pred.extend([int(p) for p in preds])

        if probs is None:
            probs = [None] * len(batch)  # type: ignore[assignment]
        for s, pred, prob in zip(batch, preds, probs):
            out_json.append({
                "id": int(s.get("id", 0)),
                "path": str(s["path"]),
                "label": int(s["label"]),
                "pred": int(pred),
                "probs": (prob if prob is not None and len(prob) > 0 else None),
            })
