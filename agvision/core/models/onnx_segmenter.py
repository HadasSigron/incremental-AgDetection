# core/models/onnx_segmenter.py
"""
ONNX segmenter loader registered for file-extension '.onnx'.
- Lazy-imports heavy deps (onnxruntime, cv2) so the module can import even if
  deps are missing; errors appear only at runtime.
- Auto-detects the required input HxW from the ONNX model and resizes accordingly.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

from .registry import register_segmenter


class OnnxSegModel:
    def __init__(self, weights: str, imgsz: Optional[int] = None, device: str = "cpu") -> None:
        # Lazy import (fail with a clear message if missing)
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(
                "Failed to import onnxruntime. On Windows ensure:\n"
                " - Microsoft Visual C++ Redistributable (x64, 2015–2022) is installed,\n"
                " - You're using 64-bit Python,\n"
                " - You installed the CPU wheel: pip install onnxruntime"
            ) from e

        self._ort = ort
        self.device = (device or "cpu").lower()

        providers = ["CPUExecutionProvider"] if self.device != "gpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(weights, providers=providers)

        # Determine model input name(s)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]

        # ---- Auto-detect required HxW from ONNX input shape (N,C,H,W) if static
        in0 = self.sess.get_inputs()[0]
        shp = list(in0.shape)  # e.g., [1, 3, 512, 512] or [1, 3, 'h', 'w']
        def _to_int(x: Any) -> Optional[int]:
            return int(x) if isinstance(x, int) else None

        h = _to_int(shp[2]) if len(shp) >= 4 else None
        w = _to_int(shp[3]) if len(shp) >= 4 else None

        if h and w:
            # Static shape in the model — use it, ignore cfg.imgsz
            self.target_hw: Tuple[int, int] = (h, w)
        else:
            # Dynamic axes — fallback to cfg-provided imgsz or 640
            side = int(imgsz or 640)
            self.target_hw = (side, side)

    # ------------- preprocessing / postprocessing -------------

    def _preprocess(self, path: str) -> Tuple["np.ndarray", Tuple[int, int]]:
        """Read image (BGR), convert to RGB, resize to model-required size, NCHW float32 [0,1]."""
        try:
            import cv2
        except Exception as e:
            raise RuntimeError("Failed to import opencv-python. Install with: pip install opencv-python") from e

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        h0, w0 = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        th, tw = self.target_hw
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
        return x, (h0, w0)

    @staticmethod
    def _resize_mask(mask: "np.ndarray", hw: Tuple[int, int]) -> "np.ndarray":
        """Resize a binary mask back to original HxW using nearest-neighbor."""
        import cv2
        h, w = hw
        m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    def _decode_yolov8_seg(self, outputs: List["np.ndarray"], orig_hw: Tuple[int, int]) -> "np.ndarray":
        """
        TODO: Implement real YOLOv8-seg postprocess (protos + coeffs -> masks).
        For now return zeros to validate the end-to-end pipeline.
        """
        h, w = orig_hw
        return np.zeros((h, w), dtype=np.uint8)

    # ------------- inference API -------------

    def predict_batch(self, paths: List[str]) -> List["np.ndarray"]:
        masks: List["np.ndarray"] = []
        for p in paths:
            x, hw = self._preprocess(p)
            outs = self.sess.run(self.out_names, {self.in_name: x})
            m = self._decode_yolov8_seg(outs, hw)
            masks.append(m)
        return masks


@register_segmenter(".onnx")
def load_onnx_segmenter(cfg: Dict[str, Any]) -> OnnxSegModel:
    """
    Loader entry registered for '.onnx' files.

    Expected cfg keys:
      - model_path: str
      - device: 'cpu'|'gpu' (optional; default 'cpu')
      - imgsz: int (optional; ignored if the model has static H/W)
    """
    weights = str(cfg.get("model_path", "")).strip()
    if not weights:
        raise ValueError("model_path is required for .onnx segmenter.")

    # If the ONNX has a static input (e.g., 512x512), the model will detect it
    # and override imgsz; otherwise we use this imgsz (or 640 by default).
    imgsz = cfg.get("imgsz", None)
    device = str(cfg.get("device", "cpu") or "cpu").lower()
    return OnnxSegModel(weights=weights, imgsz=imgsz, device=device)
