# core/models/onnx_classifier.py
"""
ONNX image classifier with robust I/O detection and safe batching.

- Detects input/output names, layout (NCHW/NHWC) and static dims from the ONNX.
- If batch dimension is static (e.g., 1), we automatically chunk inference to size 1.
- If spatial dims are static (e.g., 224x224), we override requested imgsz accordingly.
- Returns (predictions, probs) where probs are softmax over logits.

Registered as a loader via @register_classifier(".onnx").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .registry import register_classifier  # decorator


# Optional deps are loaded lazily to keep install light.
try:
    import onnxruntime as ort  # type: ignore
except Exception as e:  # pragma: no cover
    ort = None
    _ORT_IMPORT_ERROR = e

try:
    from PIL import Image  # type: ignore
except Exception as e:  # pragma: no cover
    Image = None
    _PIL_IMPORT_ERROR = e


@dataclass
class _OnnxIO:
    input_name: str
    output_name: str
    layout: str                    # "NCHW" or "NHWC"
    static_batch: Optional[int]    # e.g., 1 if model locks batch=1; None if dynamic
    target_h: Optional[int]        # spatial height if static
    target_w: Optional[int]        # spatial width if static


class OnnxClassifier:
    """Minimal ONNX classifier wrapper with batch inference."""

    def __init__(self, model_path: str | Path, imgsz: int = 224, device: str = "cpu",
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float]  = (0.229, 0.224, 0.225),
                 return_probs: bool = True) -> None:
        if ort is None:
            raise RuntimeError(
                f"onnxruntime is not installed ({_ORT_IMPORT_ERROR if '_ORT_IMPORT_ERROR' in globals() else ''}). "
                "Install 'onnxruntime' or 'onnxruntime-gpu'."
            )
        if Image is None:
            raise RuntimeError(
                f"Pillow is not installed ({_PIL_IMPORT_ERROR if '_PIL_IMPORT_ERROR' in globals() else ''}). "
                "Install 'Pillow'."
            )

        model_path = Path(str(model_path)).as_posix()
        providers = ["CPUExecutionProvider"]
        if device and "cuda" in device.lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.io = self._detect_io()
        # If model has static target size, use it; else use imgsz from cfg
        self.imgsz = int(self.io.target_h or self.io.target_w or imgsz)
        self.return_probs = bool(return_probs)

        # Stats as CHW tensors (we will reshape appropriately in NHWC path)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.asarray(std,  dtype=np.float32).reshape(3, 1, 1)

    # ---------------- internal ----------------
    def _detect_io(self) -> _OnnxIO:
        inps = self.session.get_inputs()
        outs = self.session.get_outputs()
        if not inps or not outs:
            raise RuntimeError("ONNX model must expose at least one input and one output.")

        inp = inps[0]
        out = outs[0]
        shape = list(inp.shape)  # could be ['N','C','H','W'], [1,3,224,224], etc.
        norm = [None if (isinstance(d, str) or d is None) else int(d) for d in shape]

        layout = "NCHW"
        if len(norm) == 4:
            N, A, B, C = norm
            if C == 3:
                layout = "NHWC"
            elif A == 3:
                layout = "NCHW"

        static_batch = None
        tgt_h = tgt_w = None
        if len(norm) == 4:
            if norm[0] is not None:
                static_batch = norm[0]
            if layout == "NCHW":
                tgt_h, tgt_w = norm[2], norm[3]
            else:
                tgt_h, tgt_w = norm[1], norm[2]

        return _OnnxIO(
            input_name=inp.name,
            output_name=out.name,
            layout=layout,
            static_batch=static_batch,
            target_h=tgt_h,
            target_w=tgt_w,
        )

    def _preprocess_one(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        h = int(self.io.target_h or self.imgsz)
        w = int(self.io.target_w or self.imgsz)
        img = img.resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC

        if self.io.layout == "NCHW":
            arr = arr.transpose(2, 0, 1)  # CHW
            arr = (arr - self.mean) / self.std
        else:  # NHWC
            mean = self.mean.transpose(2, 1, 0)  # (1,1,3)
            std = self.std.transpose(2, 1, 0)    # (1,1,3)
            arr = (arr - mean) / std
        return arr

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _run_chunk(self, paths: List[str]) -> tuple[list[int], list[list[float]]]:
        batch = [self._preprocess_one(p) for p in paths]
        x = np.stack(batch, axis=0)
        outs = self.session.run([self.io.output_name], {self.io.input_name: x})[0]
        if outs.ndim != 2:
            raise RuntimeError(f"Unexpected ONNX output shape: {outs.shape} (expected [N, C]).")
        probs = self._softmax(outs)
        preds = probs.argmax(axis=1).astype(int).tolist()
        return preds, probs.astype(float).tolist()

    # ---------------- public API ----------------
    def predict_batch(self, paths: List[str]) -> tuple[list[int], Optional[list[list[float]]]]:
        # If static batch is 1, chunk input accordingly
        chunk = 1 if self.io.static_batch == 1 else len(paths)
        preds_all: list[int] = []
        probs_all: list[list[float]] = []
        for i in range(0, len(paths), chunk):
            preds, probs = self._run_chunk(paths[i:i + chunk])
            preds_all.extend(preds)
            probs_all.extend(probs)
        return preds_all, probs_all


# ---------------- registry hook ----------------

@register_classifier(".onnx")
def _load_onnx_classifier(config: Dict[str, object]) -> OnnxClassifier:
    """
    Registry factory: build OnnxClassifier from a generic config dict.

    Expected keys (all optional except model_path):
      - model_path / path / weights : str
      - device : "cpu" | "cuda"
      - imgsz  : int
      - mean   : tuple/list of 3 floats
      - std    : tuple/list of 3 floats
      - return_probs : bool
    """
    mp = Path(str(config.get("model_path") or config.get("path") or config.get("weights", "")))
    if not mp.exists():
        raise FileNotFoundError(f"Classifier weights not found: {mp}")

    imgsz = int(config.get("imgsz", 224))
    device = str(config.get("device", "cpu"))
    mean = tuple(config.get("mean", (0.485, 0.456, 0.406)))  # type: ignore[arg-type]
    std  = tuple(config.get("std",  (0.229, 0.224, 0.225)))  # type: ignore[arg-type]
    return_probs = bool(config.get("return_probs", True))
    return OnnxClassifier(
        model_path=mp, imgsz=imgsz, device=device,
        mean=mean, std=std, return_probs=return_probs
    )
