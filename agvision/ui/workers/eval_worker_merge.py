# ui/workers/eval_worker_merge.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml

from core.config.schema import AppConfig
from core.config.defaults import apply_task_defaults


def _shallow_merge(dst: Dict[str, Any], src: Optional[Dict[str, Any]]) -> None:
    """Shallow merge: only top-level keys (e.g., 'eval', 'model', 'dataset', 'paths')."""
    if not src:
        return
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k].update(v)
        else:
            dst[k] = v


def _infer_domain(yaml_path: Path, overrides: Optional[Dict[str, Any]]) -> str:
    """
    Choose a valid domain for AppConfig.
    Priority:
      1) overrides['domain'] if provided
      2) YAML 'domain' if present
      3) Infer from path 'configs/benchmarks/<domain>/*' if matches known values
      4) Fallback 'plant_disease'
    """
    if overrides and isinstance(overrides.get("domain"), str) and overrides["domain"]:
        return str(overrides["domain"])

    parts = [p.lower() for p in yaml_path.parts]
    candidates = {"weeds", "plant_disease", "fruit", "uav"}
    for i, p in enumerate(parts):
        if p == "benchmarks" and i + 1 < len(parts) and parts[i + 1] in candidates:
            return parts[i + 1]
    return "plant_disease"


def _normalize_dataset_paths(dsd: Dict[str, Any]) -> None:
    """
    Ensure DatasetConfig has a 'root' and that img_dir/mask_dir/ann_file are relative to it.

    Accepts inputs where img_dir/mask_dir/ann_file can be absolute:
    - If 'root' is missing, compute a common ancestor and rewrite subpaths relative to it.
    - If only one path exists, set root to its parent and make that field relative.
    """
    img = dsd.get("img_dir")
    ann = dsd.get("ann_file") or dsd.get("annotations")  # accept alias here too
    msk = dsd.get("mask_dir")

    img_p = Path(str(img)).expanduser() if img else None
    ann_p = Path(str(ann)).expanduser() if ann else None
    msk_p = Path(str(msk)).expanduser() if msk else None

    if dsd.get("root"):
        return

    candidates = [p for p in (img_p, ann_p, msk_p) if p is not None]
    if not candidates:
        raise ValueError("dataset.root is missing and no img_dir/ann_file/mask_dir to infer from.")

    common = Path(os.path.commonpath([str(p.resolve()) for p in candidates]))
    dsd["root"] = common

    if img_p:
        try:
            dsd["img_dir"] = img_p.resolve().relative_to(common)
        except Exception:
            dsd["img_dir"] = img_p
    if msk_p:
        try:
            dsd["mask_dir"] = msk_p.resolve().relative_to(common)
        except Exception:
            dsd["mask_dir"] = msk_p
    if ann_p:
        try:
            dsd["ann_file"] = ann_p.resolve().relative_to(common)
        except Exception:
            dsd["ann_file"] = ann_p


def build_appconfig(
    benchmark_yaml: str,
    auto_model: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """
    Build a strongly-typed AppConfig from:
      - Benchmark YAML (required)
      - Auto-detected model dict (optional, preferred)
      - UI overrides (optional)
      - Back-compat: model_name for 'script' plugins

    Merge precedence:
      benchmark < auto_model < overrides

    Normalization done here so schema validation will pass:
      - benchmark_name := YAML.name (or file stem) if missing
      - domain := overrides.domain or infer from path or default
      - dataset.root inferred from img_dir/mask_dir/ann_file if missing
      - model.path -> model.weights
      - model.format normalized to canonical values (pytorch/onnx/torchscript/tensorrt/timm)
    """
    ypath = Path(benchmark_yaml)
    if not ypath.exists() or not ypath.is_file():
        raise FileNotFoundError(f"Benchmark YAML not found: {ypath}")

    data: Dict[str, Any] = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Benchmark YAML must define a mapping (dict-like).")

    # Ensure structural keys exist
    data.setdefault("dataset", {})
    data.setdefault("model", {})

    # benchmark_name
    if not data.get("benchmark_name"):
        data["benchmark_name"] = data.get("name") or ypath.stem

    # domain
    if not data.get("domain"):
        data["domain"] = _infer_domain(ypath, overrides)

    # dataset root normalization (fills dataset.root when missing)
    if isinstance(data["dataset"], dict):
        _normalize_dataset_paths(data["dataset"])

    # merge auto-model + overrides
    _shallow_merge(data, {"model": dict(auto_model or {})})
    _shallow_merge(data, overrides)

    # normalize model section
    model = data["model"] or {}
    if "path" in model and "weights" not in model:
        model["weights"] = model.pop("path")

    # if format still missing, default to pytorch if weights look like .pt/.pth; else 'script'
    if not model.get("format"):
        w = str(model.get("weights", "")).lower()
        model["format"] = "pytorch" if (w.endswith(".pt") or w.endswith(".pth")) else "script"

    # canonicalize format aliases to enum-friendly values
    fmt = str(model.get("format", "")).strip().lower()
    alias = {
        "torch/pytorch": "pytorch",
        "pytorch":       "pytorch",
        "torch":         "pytorch",
        "pt":            "pytorch",
        "ts":            "torchscript",
    }
    model["format"] = alias.get(fmt, fmt)

    # back-compat: fill name for script-like plugins when provided as 'model_name'
    if model.get("format") == "script" and not model.get("name") and model_name:
        model["name"] = str(model_name)

    data["model"] = model

    # Build & apply per-task defaults
    cfg = AppConfig(**data)
    cfg = apply_task_defaults(cfg)
    return cfg
