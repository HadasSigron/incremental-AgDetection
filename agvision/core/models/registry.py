# core/models/registry.py
from __future__ import annotations
from typing import Callable, Dict, Any
from pathlib import Path

_DETECTOR_LOADERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
_CLASSIFIER_LOADERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
_SEGMENTER_LOADERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

def register_detector(kind: str):
    def deco(fn: Callable[[Dict[str, Any]], Any]):
        _DETECTOR_LOADERS[kind] = fn
        return fn
    return deco

def register_classifier(kind: str):
    def deco(fn: Callable[[Dict[str, Any]], Any]):
        _CLASSIFIER_LOADERS[kind] = fn
        return fn
    return deco

def register_segmenter(kind: str):
    def deco(fn: Callable[[Dict[str, Any]], Any]):
        _SEGMENTER_LOADERS[kind] = fn
        return fn
    return deco


# -------- autoload built-in plugins (so decorators run) --------
def _autoload_builtin_plugins() -> None:
    """
    Import built-in model loader modules so their @register_* decorators run.
    Keep imports inside try/except — optional dependencies may not be installed yet.
    """
    # Simulators (if your project provides them) — safe if missing
    try:
        from . import simulate  # registers "simulate" for multiple types (if implemented)
        _ = simulate
    except Exception as e:
        print(f"[models.registry] WARN: failed to autoload simulate: {e}")

    # Classifier backends
    try:
        from . import onnx_classifier  # registers ".onnx" for classifiers
        _ = onnx_classifier
    except Exception as e:
        print(f"[models.registry] WARN: failed to autoload onnx_classifier: {e}")

    # Segmenter backends
    try:
        from . import onnx_segmenter  # registers ".onnx" for segmenters
        _ = onnx_segmenter
    except Exception as e:
        print(f"[models.registry] WARN: failed to autoload onnx_segmenter: {e}")


# Run once on module import
_autoload_builtin_plugins()


# -------------------------- loaders API --------------------------
def load_detector(config: Dict[str, Any]) -> Any:
    _autoload_builtin_plugins()

    if bool(config.get("simulate", False)) and "simulate" in _DETECTOR_LOADERS:
        return _DETECTOR_LOADERS["simulate"](config)

    mp = str(config.get("model_path", "")).strip()
    if not mp:
        raise RuntimeError("model_path is required unless simulate=True or predictions_json is provided.")
    ext = Path(mp).suffix.lower()
    if ext in _DETECTOR_LOADERS:
        return _DETECTOR_LOADERS[ext](config)
    raise RuntimeError(f"Unsupported detector format: {ext}. Register a loader via register_detector().")


def load_classifier(config: Dict[str, Any]) -> Any:
    _autoload_builtin_plugins()

    if bool(config.get("simulate", False)) and "simulate" in _CLASSIFIER_LOADERS:
        return _CLASSIFIER_LOADERS["simulate"](config)

    mp = str(config.get("model_path", "")).strip()
    if not mp:
        raise RuntimeError("model_path is required unless simulate=True or predictions_json is provided.")
    ext = Path(mp).suffix.lower()
    if ext in _CLASSIFIER_LOADERS:
        return _CLASSIFIER_LOADERS[ext](config)
    raise RuntimeError(f"Unsupported classifier format: {ext}. Register a loader via register_classifier().")


def load_segmenter(config: Dict[str, Any]) -> Any:
    _autoload_builtin_plugins()

    if bool(config.get("simulate", False)) and "simulate" in _SEGMENTER_LOADERS:
        return _SEGMENTER_LOADERS["simulate"](config)

    mp = str(config.get("model_path", "")).strip()
    if not mp:
        raise RuntimeError("model_path is required unless simulate=True or predictions_json is provided.")
    ext = Path(mp).suffix.lower()
    if ext in _SEGMENTER_LOADERS:
        return _SEGMENTER_LOADERS[ext](config)
    raise RuntimeError(f"Unsupported segmenter format: {ext}. Register a loader via register_segmenter().")
