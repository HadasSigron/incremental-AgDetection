"""Lightweight runs store (directory layout for run artifacts)."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime

def make_run_dir(cfg) -> Path:
    """
    Create the canonical run directory under cfg.paths.runs_root:
      <runs_root>/<task>/<domain>/<benchmark>/<timestamp>
    Note:
      cfg.task/cfg.domain can be Enums or strings; we normalize to lowercase.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task   = getattr(cfg.task, "value", str(cfg.task)).lower()
    domain = getattr(cfg.domain, "value", str(cfg.domain)).lower()
    bench  = str(cfg.benchmark_name)
    base = Path(cfg.paths.runs_root).expanduser().resolve()
    run_dir = base / task / domain / bench / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
