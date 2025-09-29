"""Incremental evaluation orchestration."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict

from .hashing import sample_key, dict_sig, model_sig
from .store import CacheStore
from .reporter import save_predictions, save_summary
from .runs import make_run_dir

def evaluate_incremental(cfg, dataset, runner, cancel=None) -> Dict[str, Any]:
    """
    Run evaluation with incremental caching.
    Args:
      cfg: AppConfig (must include paths.runs_root, model.weights, etc.)
      dataset: adapter exposing iter_samples(split="val")
      runner: object exposing infer_one(sample) -> dict (a single-sample prediction)
      cancel: optional CancelToken (supports graceful stop)
    Returns:
      dict of metrics (placeholder here; replace with real computation).
    """
    # Create a unique directory for this run
    run_dir = make_run_dir(cfg)

    # Base signatures used by all samples in this run
    cfg_sig = dict_sig(_cfg_to_dict(cfg))
    model_w = getattr(cfg.model, "weights", None)
    model_sig_ = model_sig(model_w)

    # Persistent cache lives under runs_root
    cache = CacheStore(Path(cfg.paths.runs_root) / ".cache.sqlite")
    preds: list[dict] = []

    # Main loop over evaluation split (default "val")
    for smp in dataset.iter_samples("val"):
        if cancel and getattr(cancel, "is_set", lambda: False)():
            break

        key = sample_key(smp["path"], model_sig_, cfg_sig)
        cached = cache.get(key)
        if cached:
            preds.append(cached)
            continue

        pred = runner.infer_one(smp)  # Real model inference for this sample
        cache.put(key, pred)
        preds.append(pred)

    # TODO: replace with actual metrics computation per task
    metrics: Dict[str, Any] = {"status": "ok", "num_predictions": len(preds)}

    # Persist artifacts for this run
    save_predictions(run_dir, preds)
    save_summary(run_dir, metrics, _cfg_to_dict(cfg))
    return metrics

def _cfg_to_dict(cfg):
    """
    Convert AppConfig to a plain dict without depending on a specific
    Pydantic major version. Falls back to a generic JSON conversion.
    """
    try:
        return cfg.model_dump()           # pydantic v2
    except Exception:
        try:
            return cfg.dict()             # pydantic v1
        except Exception:
            return json.loads(json.dumps(cfg, default=str))
