# agvision/core/models/__init__.py
from __future__ import annotations
from .registry import register_detector, register_classifier

# --- import wrappers ---
from .simulate import SimulatedDetector                 # detection (simulate)
from .yolo import YoloDetector                          # detection (.pt YOLO)
from .torchscript import TorchscriptClassifier, SimulatedClassifier  # classification (.pt/.ts + simulate)

# ------- detectors -------
@register_detector("simulate")
def _load_sim_detector(cfg):
    return SimulatedDetector()

@register_detector(".pt")
def _load_yolo_pt(cfg):
    return YoloDetector(
        cfg["model_path"],
        conf=float(cfg.get("conf_threshold", 0.25)),
        iou=float(cfg.get("iou_threshold", 0.45)),
        imgsz=int(cfg.get("imgsz", 640)),
        device=cfg.get("device"),   # "cuda"/"cpu"/None
        half=bool(cfg.get("half", False)),
    )

# ------- classifiers -------
@register_classifier("simulate")
def _load_sim_classifier(cfg):
    n = int(cfg.get("n_classes") or cfg.get("num_classes") or 1000)
    return SimulatedClassifier(n_classes=n)

@register_classifier(".pt")
@register_classifier(".ts")
def _load_ts_classifier(cfg):
    return TorchscriptClassifier(
        cfg["model_path"],
        device=str(cfg.get("device", "cpu")),
        imgsz=int(cfg.get("imgsz", 224)),
    )
