from __future__ import annotations

import json
import time
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from PIL import Image

from core.runners.base import BaseRunner
from core.config.schema import AppConfig, ModelFormat
from core.models.registry import load_segmenter
from core.types.runner_result import RunnerResult, ensure_runner_result
from core.metrics.segmentation import compute_metrics_from_config



from core.eval.store import make_run_dir
from core.eval.reporter import save_predictions, save_summary

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

    Expected per-sample keys (segmentation):
      {
        "id": int,
        "path": str,                         # input image path
        # optional ground-truth:
        "mask_path" or "gt_mask_path": str  # ground-truth mask file path
      }
    """
    if hasattr(dataset, "iter_samples"):
        return dataset.iter_samples()
    return iter(dataset)


def _out_dir(cfg: AppConfig) -> Path:
    """
    Resolve the output directory:

    Priority:
      1) Respect cfg.eval.params['out_dir'] if provided.
      2) Try reporter.make_run_dir() with modern or legacy signatures.
      3) Fallback to: <runs_root>/<Task>/<domain>/<benchmark>/<timestamp-like stub>

    Always avoids double 'runs/' and ensures the directory exists.
    """
    params = getattr(cfg.eval, "params", {}) if hasattr(cfg, "eval") else {}
    user_out = params.get("out_dir")
    if user_out:
        p = Path(str(user_out))
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- Try reporter.make_run_dir with multiple signatures ----
    rel: Path | None = None
    try:
        # Newer API: make_run_dir(task, domain, benchmark)
        rel = Path(make_run_dir("Segmentation", cfg.domain.value, cfg.benchmark_name))  # type: ignore[arg-type]
    except TypeError:
        try:
            # Legacy API: make_run_dir() returns a relative path
            rel = Path(make_run_dir())  # type: ignore[misc]
        except Exception:
            rel = None

    # ---- Fallback if reporter API is not as expected ----
    if rel is None:
        # Minimal deterministic fallback (no real timestamp; fine for tests)
        rel = Path("runs") / "Segmentation" / str(cfg.domain.value) / str(cfg.benchmark_name) / "test"

    # Avoid "runs/runs/*" if reporter already prefixed it
    if not rel.is_absolute() and rel.parts and rel.parts[0].lower() == "runs":
        rel = Path(*rel.parts[1:])

    out = rel if rel.is_absolute() else (cfg.paths.runs_root / rel)
    out.mkdir(parents=True, exist_ok=True)
    return out



def _dynamic_import(script_path: Path) -> Any:
    """
    Import a Python module from a filesystem path.

    The module is expected to expose a callable entrypoint (default: 'build')
    that returns a segmenter implementing either:
      - predict_batch(paths: List[str]) -> List[MaskLike]
      - predict(path: str) -> MaskLike

    MaskLike:
      - str | Path                         # path to an EXISTING saved mask file
      - np.ndarray (H,W) or (H,W,1)       # in-memory mask
      - PIL.Image.Image                   # in-memory mask
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


def _build_segmenter_from_script(cfg: AppConfig) -> Any:
    """
    Build a segmenter from a user script.

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


def _load_segmenter(cfg: AppConfig) -> Any:
    """
    Build/Load a segmenter according to cfg.model.format.

    Supported:
      - script: dynamic plugin file (models/<name>.py)
      - onnx/tensorrt/...: via core.models.registry.load_segmenter()
    """
    fmt = cfg.model.format

    if fmt == ModelFormat.script:
        return _build_segmenter_from_script(cfg)

    # Registry-backed loaders (e.g., '.onnx' -> core.models.onnx_segmenter)
    weights = (getattr(cfg.model, "weights", None) or cfg.model.path)
    weights = Path(str(weights)).as_posix()

    device = (cfg.eval.device or "cpu").lower()
    imgsz = int(getattr(cfg.model, "imgsz", getattr(cfg.dataset, "input_size", 640)) or 640)

    # Thresholds: model.* overrides eval.params.*
    params = dict(getattr(cfg.eval, "params", {}) or {})
    conf_thr = float(getattr(cfg.model, "conf_threshold", params.get("conf_threshold", 0.25)) or 0.25)
    iou_thr  = float(getattr(cfg.model, "iou_threshold",  params.get("iou_threshold",  0.50)) or 0.50)
    mask_thr = float(getattr(cfg.model, "mask_threshold", params.get("mask_threshold", 0.50)) or 0.50)

    return load_segmenter({
        "model_path": weights,
        "device": device,
        "imgsz": imgsz,
        "conf_threshold": conf_thr,
        "iou_threshold": iou_thr,
        "mask_threshold": mask_thr,
    })


def _to_uint8_mask(arr: np.ndarray) -> np.ndarray:
    """Convert a numeric mask to uint8 [0..255] for safe PNG writing."""
    if arr.dtype == np.bool_:
        return (arr.astype(np.uint8) * 255)
    uniq = np.unique(arr)
    if uniq.size <= 2 and set(uniq.tolist()).issubset({0, 1}):
        return (arr.astype(np.uint8) * 255)
    return np.clip(arr.astype(np.int32), 0, 255).astype(np.uint8)


def _save_mask(mask: Union[str, Path, np.ndarray, Image.Image], dst_dir: Path, stem: str) -> Path:
    """
    Persist (or reference) a predicted mask on disk and return its path.
    Accepts str/Path (already on disk), np.ndarray (H,W)/(H,W,1), or PIL.Image.
    """
    if isinstance(mask, (str, Path)):
        return Path(mask)

    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / f"{stem}.png"

    if isinstance(mask, np.ndarray):
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask array, got shape={mask.shape}")
        Image.fromarray(_to_uint8_mask(mask)).save(out)
        return out

    if isinstance(mask, Image.Image):
        mask.save(out)
        return out

    raise TypeError(f"Unsupported mask type: {type(mask)!r}")


# --------------------------------- runner --------------------------------------

class SegmenterRunner(BaseRunner):
    """
    AppConfig-native segmentation runner.

    Metrics policy:
      - Computes exactly the metrics listed in cfg.eval.metrics via compute_metrics_from_config.
      - If cfg.eval.metrics is empty/missing -> ValueError (defaults live in defaults.py).

    Plugin flexibility:
      - Model may return mask file paths OR in-memory masks (np.ndarray/PIL).
      - Runner normalizes by saving masks when needed and writes predictions.json.
    """

    def run(self, dataset: Any, cfg: AppConfig, cancel_token: Any | None = None) -> RunnerResult:
        t0 = time.time()
        out_dir = _out_dir(cfg)
        preds_path = out_dir / "predictions.json"
        masks_dir = out_dir / "masks"

        # --------- (A) Eval-only: use precomputed predictions ----------
        pred_json_path = getattr(cfg.eval, "params", {}).get("predictions_json")
        if pred_json_path:
            preds = json.loads(Path(str(pred_json_path)).read_text(encoding="utf-8"))

            # Build id -> GT path map from dataset
            id_to_gt: Dict[int, Optional[Path]] = {}
            for s in _iter_samples(dataset):
                sid = int(s.get("id", 0))
                gt = s.get("mask_path") or s.get("gt_mask_path")
                id_to_gt[sid] = Path(str(gt)) if gt else None

            # Align pairs (only where GT exists)
            y_true_paths: List[Path] = []
            y_pred_paths: List[Path] = []
            for p in preds:
                sid = int(p.get("id", 0))
                gt = id_to_gt.get(sid)
                pr = Path(str(p.get("mask"))) if p.get("mask") else None
                if gt is not None and pr is not None:
                    y_true_paths.append(gt)
                    y_pred_paths.append(pr)

            requested = list(getattr(cfg.eval, "metrics", []) or [])
            if not requested:
                raise ValueError("cfg.eval.metrics must be a non-empty list (e.g., ['miou','pixel_accuracy']).")
            extra = dict(getattr(cfg.eval, "params", {}) or {})

            # IMPORTANT: pass params=extra (do not unpack)
            metrics = compute_metrics_from_config(
                y_true_paths, y_pred_paths, requested=requested, params=extra
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
        model = _load_segmenter(cfg)
        bs = int(getattr(cfg.eval, "batch_size", 1) or 1)

        all_preds: List[Dict[str, Any]] = []
        y_true_paths: List[Path] = []
        y_pred_paths: List[Path] = []

        batch: List[Dict[str, Any]] = []
        for sample in _iter_samples(dataset):
            if _is_cancelled(cancel_token):
                break
            batch.append(sample)
            if len(batch) >= bs:
                self._flush_batch(model, batch, masks_dir, all_preds, y_true_paths, y_pred_paths)
                batch.clear()

        if batch and not _is_cancelled(cancel_token):
            self._flush_batch(model, batch, masks_dir, all_preds, y_true_paths, y_pred_paths)

        preds_path.write_text(json.dumps(all_preds, ensure_ascii=False), encoding="utf-8")

        requested = list(getattr(cfg.eval, "metrics", []) or [])
        if not requested:
            raise ValueError("cfg.eval.metrics must be a non-empty list (e.g., ['miou','pixel_accuracy']).")
        extra = dict(getattr(cfg.eval, "params", {}) or {})

        # IMPORTANT: pass params=extra (not **extra)
        metrics = compute_metrics_from_config(
            y_true_paths, y_pred_paths, requested=requested, params=extra
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
    def _flush_batch(
        model: Any,
        batch: List[Dict[str, Any]],
        masks_dir: Path,
        out_json: List[Dict[str, Any]],
        y_true_paths: List[Path],
        y_pred_paths: List[Path],
    ) -> None:
        """
        Run a batch, normalize predictions to file paths, collect GT/pred pairs,
        and append rows to predictions.json buffer.
        """
        paths = [str(s["path"]) for s in batch]

        # run model
        if hasattr(model, "predict_batch"):
            masks = list(model.predict_batch(paths))  # type: ignore[assignment]
        else:
            masks = [model.predict(p) for p in paths]  # type: ignore[attr-defined]

        if len(masks) != len(batch):
            raise RuntimeError("predict/predict_batch must return one mask per input.")

        # normalize to file paths and collect
        masks_dir.mkdir(parents=True, exist_ok=True)
        for s, m in zip(batch, masks):
            sid = int(s.get("id", 0))
            pred_path = _save_mask(m, masks_dir, f"{sid:08d}")  # ensures a real file path
            gt = s.get("mask_path") or s.get("gt_mask_path")

            if gt:
                y_true_paths.append(Path(str(gt)))
                y_pred_paths.append(pred_path)

            out_json.append({
                "id": sid,
                "path": str(s["path"]),
                "mask": str(pred_path),
            })
