# core/config/__init__.py
"""
Config loading API:
- load_config(path, overrides=None): YAML/TOML -> AppConfig (+defaults)
- load_config_from_dict(d): dict -> AppConfig (+defaults)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
import os
import yaml

try:
    import tomllib as toml_loader  # py3.11+
except Exception:
    toml_loader = None

from .schema import AppConfig
from .defaults import apply_task_defaults

__all__ = ["load_config", "load_config_from_dict", "AppConfig", "apply_task_defaults"]

def load_config(path: Union[str, Path], overrides: Optional[Dict[str, Any]]=None) -> AppConfig:
    data = _read_file_to_dict(Path(path))
    if overrides:
        data = _deep_merge(data, overrides)
    data = _expand_env_in_dict(data)
    return load_config_from_dict(data)

def load_config_from_dict(d: Dict[str, Any]) -> AppConfig:
    cfg = AppConfig(**d)              # schema validation
    cfg = apply_task_defaults(cfg)    # fill task-aware defaults
    return cfg

def _read_file_to_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yml", ".yaml"):
        return yaml.safe_load(text) or {}
    if path.suffix.lower() == ".toml":
        if toml_loader is None:
            raise RuntimeError("TOML not available (need Python 3.11+).")
        return toml_loader.loads(text)
    raise ValueError("Unsupported config format (use .yaml/.yml/.toml).")

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _expand_env_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    def _x(v):
        if isinstance(v, str):
            return Path(os.path.expandvars(os.path.expanduser(v))).as_posix()
        if isinstance(v, dict):
            return {k: _x(v2) for k, v2 in v.items()}
        if isinstance(v, list):
            return [_x(i) for i in v]
        return v
    return {k: _x(v) for k, v in d.items()}
