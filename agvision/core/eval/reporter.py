"""Reporter for saving artifacts."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def save_predictions(run_dir: Path | str, preds: list) -> str:
    out = Path(run_dir) / "predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    return str(out)

def save_summary(run_dir: Path | str, metrics: Dict[str, Any], config: Dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config.used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
