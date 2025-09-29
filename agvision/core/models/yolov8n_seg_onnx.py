"""ONNX classifier wrapper (placeholder)."""


# models/yolov8n_seg_onnx.py
"""
Minimal ONNX segmenter plugin for the 'script' model format.

This is a thin wrapper around onnxruntime. It handles:
- session creation (CPU by default),
- basic preprocessing (resize, BGR->RGB, NHWC->NCHW, [0..1] float),
- batch prediction API: predict_batch(paths) -> list[Mask or None]

IMPORTANT:
- YOLOv8 *segmentation* ONNX requires custom post-processing (prototypes+coeffs -> mask).
  This file includes a placeholder 'decode_yolov8_seg' with a clear error.
  Implement it based on your exact model outputs to get real masks.
"""

from __future__ import annotations
from typing import Any, List, Tuple, Optional
from pathlib import Path

import numpy as np
import cv2  # opencv-python
import onnxruntime as ort


class ORTOnnxSegmenter:
    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 640) -> None:
        self.imgsz = int(imgsz)
        sess_opts = ort.SessionOptions()
        providers = ["CPUExecutionProvider"] if device.lower() != "gpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(weights, sess_options=sess_opts, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]

    @staticmethod
    def _preprocess(path: str, imgsz: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        h0, w0 = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        # NHWC -> NCHW, float32 [0..1]
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x, (h0, w0)

    @staticmethod
    def _resize_mask(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
        """Resize binary mask (H, W) back to original image shape."""
        h, w = hw
        out = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return (out > 0).astype(np.uint8)

    def _decode_yolov8_seg(self, outputs: List[np.ndarray], orig_hw: Tuple[int, int]) -> np.ndarray:
        """
        PLACEHOLDER: implement proper YOLOv8-seg post-process here:
          - Identify det head and mask prototypes tensor,
          - For each detection: recover mask via prototypes @ coeffs,
          - Combine/argmax per class if needed,
          - Return a final binary mask (H, W) for your evaluation.

        For now, we raise with a clear message so you know to wire it.
        """
        raise NotImplementedError(
            "YOLOv8-seg ONNX postprocess is not implemented yet. "
            "Parse 'outputs' (dets + protos) and generate a binary mask per image."
        )

    def predict_batch(self, paths: List[str]) -> List[Optional[np.ndarray]]:
        """
        Contract expected by SegmenterRunner:
          - Returns a list of binary masks (H,W) with values {0,1} or None if not detected.

        NOTE:
          - This function currently calls a placeholder YOLOv8 decode. Implement it for real results.
        """
        masks: List[Optional[np.ndarray]] = []
        for p in paths:
            x, (h0, w0) = self._preprocess(p, self.imgsz)
            outs = self.sess.run(self.output_names, {self.input_name: x})
            # Try a simple case first: any output that looks like 1x1xH'xW' prob-map
            simple = None
            for y in outs:
                if y.ndim == 4 and y.shape[0] == 1 and y.shape[1] in (1,):
                    prob = 1/(1+np.exp(-y[0,0])) if y.dtype != np.float32 else y[0,0]
                    simple = (prob > 0.5).astype(np.uint8)
                    break
            if simple is not None:
                masks.append(self._resize_mask(simple, (h0, w0)))
                continue

            # Otherwise fall back to YOLOv8-seg decode (needs implementation)
            try:
                m = self._decode_yolov8_seg(outs, (h0, w0))
            except NotImplementedError as e:
                # If you prefer to see the pipeline end-to-end, return a blank mask instead of raising:
                # masks.append(np.zeros((h0, w0), dtype=np.uint8)); continue
                raise
            masks.append(m.astype(np.uint8))
        return masks


def build(cfg: Any) -> ORTOnnxSegmenter:
    """
    Entry-point required by 'script' model format.
    Reads imgsz/device/weights from cfg and returns ORTOnnxSegmenter instance.
    """
    weights = (cfg.model.weights or cfg.model.path).as_posix() if hasattr(cfg.model, "weights") else str(cfg.model.path)
    device = (cfg.eval.device or "cpu") if hasattr(cfg, "eval") else "cpu"
    imgsz = int(getattr(cfg.model, "imgsz", getattr(cfg.dataset, "input_size", 640)) or 640)
    return ORTOnnxSegmenter(weights=weights, device=device, imgsz=imgsz)
