from __future__ import annotations
from typing import List, Dict, Any, Optional

class YoloDetector:
    """
    Thin wrapper around Ultralytics YOLOv8 for object detection.
    Returns per-image detections as [[x1,y1,x2,y2,score,cls_id], ...].
    """
    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,  # "cuda" / "cpu" / None (auto)
        half: bool = False,
    ):
        from ultralytics import YOLO  # lazy import to avoid heavy import during app start
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.half = half

    def predict_batch(self, batch_samples: List[Dict[str, Any]]) -> List[List[List[float]]]:
        paths = [str(s["path"]) for s in batch_samples]
        results = self.model(
            paths,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,   # "cuda"/"cpu"/None
            half=self.half,
            verbose=False,
            stream=False,
        )
        out: List[List[List[float]]] = []
        for r in results:
            dets: List[List[float]] = []
            if getattr(r, "boxes", None) is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss  = r.boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
                    dets.append([float(x1), float(y1), float(x2), float(y2), float(sc), int(c)])
            out.append(dets)
        return out
