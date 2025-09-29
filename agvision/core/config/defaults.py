"""Default configuration values (placeholder)."""

from __future__ import annotations

# core/config/defaults.py
"""
Task-aware defaults and helpers to build/complete an AppConfig.

This module centralizes:
- Default metrics per task
- Sensible runtime defaults (batch/num_workers per task)
- A small factory to create an AppConfig skeleton for the UI
"""


from typing import Dict, List
from pathlib import Path

from .schema import (
    AppConfig, TaskType, Domain,
    DatasetConfig, DatasetKind,
    ModelConfig, ModelFormat,
    EvalConfig, PathsConfig, LoggingConfig,
)

# -------------------- Metric presets per task ----------------------------------

DEFAULT_METRICS: Dict[TaskType, List[str]] = {
    TaskType.classification: ["accuracy", "macro_f1"],
    TaskType.detection: ["map50", "map5095"],     # mAP@0.5 and mAP@[.5:.95]
    TaskType.segmentation: ["miou", "dice"],
}

# -------------------- Runtime defaults per task --------------------------------

DEFAULT_BATCH: Dict[TaskType, int] = {
    TaskType.classification: 64,
    TaskType.detection: 8,
    TaskType.segmentation: 4,
}

DEFAULT_WORKERS: Dict[TaskType, int] = {
    TaskType.classification: 4,
    TaskType.detection: 4,
    TaskType.segmentation: 4,
}


def apply_task_defaults(cfg: AppConfig) -> AppConfig:
    """
    Fill missing evaluation fields with task-aware defaults:
    - metrics: if empty, use DEFAULT_METRICS[task]
    - batch_size / num_workers: if not provided (<=0), use task defaults
    - device 'auto': pick 'cuda' if available, else 'cpu'
    """
    # Metrics
    if not cfg.eval.metrics:
        cfg.eval.metrics = list(DEFAULT_METRICS[cfg.task])

    # Batch/Workers
    if cfg.eval.batch_size is None or cfg.eval.batch_size <= 0:
        cfg.eval.batch_size = DEFAULT_BATCH[cfg.task]
    if cfg.eval.num_workers is None or cfg.eval.num_workers < 0:
        cfg.eval.num_workers = DEFAULT_WORKERS[cfg.task]

    # Device fallback
    if cfg.eval.device == "auto":
        cfg.eval.device = "cuda" if _cuda_available() else "cpu"

    return cfg


def _cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433 (runtime import)
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def make_skeleton(
    task: TaskType,
    *,
    domain: Domain = Domain.plant_disease,
    benchmark_name: str = "UNKNOWN_BENCHMARK",
    dataset_root: Path = Path("data/benchmarks/UNKNOWN"),
) -> AppConfig:
    """
    Build a minimal AppConfig skeleton for a given task with sensible defaults.
    Callers (UI/server) can modify fields and then pass into the runner.

    The dataset.kind is inferred from task, model defaults to a 'timm' placeholder.
    """
    if task == TaskType.classification:
        ds_kind = DatasetKind.classification_folder
    elif task == TaskType.detection:
        ds_kind = DatasetKind.coco_detection
    else:
        ds_kind = DatasetKind.masks_segmentation

    cfg = AppConfig(
        task=task,
        domain=domain,
        benchmark_name=benchmark_name,
        dataset=DatasetConfig(
            kind=ds_kind,
            root=dataset_root,
            split="test",
            input_size=224,
        ),
        model=ModelConfig(
            format=ModelFormat.timm,
            name="resnet50",  # placeholder for UI; adapters may override
            num_classes=None,
            device="auto",
        ),
        eval=EvalConfig(
            batch_size=DEFAULT_BATCH[task],
            num_workers=DEFAULT_WORKERS[task],
            device="auto",
            mixed_precision=False,
            seed=42,
            metrics=list(DEFAULT_METRICS[task]),
            params={}
        ),
        paths=PathsConfig(),
        logging=LoggingConfig(),
    )
    return cfg
