# agvision/core/runners/detector_runner.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.runners.base import BaseRunner
from core.config.schema import AppConfig, ModelFormat
from core.types.runner_result import RunnerResult, ensure_runner_result
from core.eval.store import make_run_dir                    # << כאן השינוי
from core.eval.reporter import save_predictions, save_summary
from core.metrics.detection import compute_metrics_from_config  # <-- like classifier runner

# optional registry loader
try:
    from core.models.registry import load_detector as _registry_load_detector  # noqa: F401
except Exception:
    _registry_load_detector = None  # type: ignore[misc]


# ------------------------------- helpers --------------------------------------

def _is_cancelled(token: Any | None) -> bool:
    """Best-effort cancellation check."""
    try:
        return bool(token and token.is_set())
    except Exception:
        return bool(token)


def _iter_samples(dataset: Any) -> Iterable[Dict[str, Any]]:
    """Adapter-agnostic iteration over samples/images."""
    if hasattr(dataset, "iter_samples"):
        return dataset.iter_samples()
    if hasattr(dataset, "iter_images"):
        return dataset.iter_images()
    return iter(dataset)


def _image_ids(dataset: Any) -> List[int]:
    """Resolve image ids for metrics selection/filtering."""
    if hasattr(dataset, "image_ids"):
        xs = dataset.image_ids()
        return xs if isinstance(xs, list) else list(xs)
    if hasattr(dataset, "coco_gt"):
        try:
            return list(dataset.coco_gt().getImgIds())
        except Exception:
            return []
    try:
        return [int(s["id"]) for s in _iter_samples(dataset)]
    except Exception:
        return []


def _out_dir(cfg: AppConfig) -> Path:
    """Resolve run output directory using store.make_run_dir(cfg)."""
    return make_run_dir(cfg)


def _dynamic_detector_from_script(cfg: AppConfig):
    """Build a detector from a user script (models/<name>.py or explicit script_path)."""
    from importlib.util import spec_from_file_location, module_from_spec
    extra = dict(cfg.model.extra or {})
    script_path = Path(str(extra.get("script_path") or (Path("models") / f"{cfg.model.name}.py")))
    spec = spec_from_file_location(script_path.stem, script_path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import detector script: {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    entry = getattr(module, str(extra.get("entrypoint", "build")))
    return entry(cfg)


def _build_detector(cfg: AppConfig):
    """Build/Load a detector according to cfg.model.format and optional registry."""
    if cfg.model.format == ModelFormat.script:
        return _dynamic_detector_from_script(cfg)
    if _registry_load_detector is not None:
        params = getattr(cfg.eval, "params", {}) or {}
        w = getattr(cfg.model, "weights", None)
        return _registry_load_detector({
            "model_path": (w.as_posix() if w else ""),
            "device": cfg.eval.device,
            "imgsz": int(getattr(cfg.model, "imgsz", 640) or 640),
            "conf_threshold": float(params.get("conf_threshold", 0.25)),
            "iou_threshold": float(params.get("iou_threshold", 0.45)),
        })
    raise NotImplementedError(f"No detector loader for format={cfg.model.format.value!r}.")


def _flush_batch(model: Any, batch: List[Dict[str, Any]],
                 metas: List[Dict[str, Any]], out: List[Dict[str, Any]]) -> None:
    """
    Run a batch through the detector (batch-only API) and append COCO-format predictions.

    Model contract:
      predict_batch(batch) -> List[List[(x1, y1, x2, y2, score, cls_id)]]
        - returns a list of detections per input sample
    """
    out.extend(
        {
            "image_id": int(meta["id"]),
            "category_id": int(cls_id),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        }
        for dets, meta in zip(model.predict_batch(batch), metas)
        for (x1, y1, x2, y2, score, cls_id) in dets
    )


# --------------------------------- runner --------------------------------------

class DetectorRunner(BaseRunner):
    """
    Detector runner aligned with classifier runner:

    Modes:
      (A) Eval-only:
          If cfg.eval.params['predictions_json'] is provided, skip inference and
          compute metrics strictly as requested by cfg.eval.metrics.

      (B) Inference:
          Otherwise build a detector and run batched inference (batch-only API).

    Outputs:
      - metrics: computed strictly by the list in cfg.eval.metrics
      - artifacts:
          * predictions_json (COCO-style)
          * run_dir
    """

    def run(self, dataset: Any, cfg: AppConfig, cancel_token: Any | None = None) -> RunnerResult:
        run_dir = _out_dir(cfg)
        img_ids = _image_ids(dataset)

        params = getattr(cfg.eval, "params", {}) or {}
        requested = list(cfg.eval.metrics)             # must exist (defaults.py)
        strict = bool(params.get("metrics_strict", True))

        preds_cfg = params.get("predictions_json")
        if preds_cfg:
            preds_path = str(preds_cfg)
        else:
            model = _build_detector(cfg)

            # Enforce batch-only API (no fallback to predict(path))
            if not hasattr(model, "predict_batch"):
                raise TypeError("Detector model must implement predict_batch(batch) -> list-of-detections per sample.")

            bs = int(getattr(cfg.eval, "batch_size", 1) or 1)
            preds_coco: List[Dict[str, Any]] = []
            batch: List[Dict[str, Any]] = []
            metas: List[Dict[str, Any]] = []

            for sample in _iter_samples(dataset):
                if _is_cancelled(cancel_token):
                    break
                metas.append({"id": int(sample["id"])})
                batch.append(sample)
                if len(batch) >= bs:
                    _flush_batch(model, batch, metas, preds_coco)
                    batch.clear()
                    metas.clear()

            if batch and not _is_cancelled(cancel_token):
                _flush_batch(model, batch, metas, preds_coco)

            preds_path = save_predictions(run_dir, preds_coco)

        # Compute detection metrics strictly by config (in metrics package)
        metrics = compute_metrics_from_config(
            dataset, preds_path, img_ids, requested=requested, strict=strict
        )

        # Persist summary (optional but useful; keeps parity with existing flows)
        save_summary(run_dir, metrics, {
            "task": cfg.task.value,
            "domain": cfg.domain.value,
            "benchmark": cfg.benchmark_name,
        })

        return ensure_runner_result({
            "metrics": metrics,
            "artifacts": {
                "predictions_json": preds_path,
                "run_dir": str(run_dir),
            },
        })
