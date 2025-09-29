from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import hashlib

class SimulatedDetector:
    """Deterministic fake detector for quick testing and demos."""
    def predict_batch(self, batch_samples: List[Dict[str, Any]]) -> List[List[List[float]]]:
        out: List[List[List[float]]] = []
        for s in batch_samples:
            h = int(hashlib.sha1(str(Path(s["path"])).encode()).hexdigest(), 16)
            n = (h % 2) + 1  # 1-2 boxes per image
            dets: List[List[float]] = []
            for i in range(n):
                x1 = 30 + (h % 50)
                y1 = 40 + ((h >> 3) % 60)
                x2, y2 = x1 + 60, y1 + 40
                score  = 0.6 + ((h >> (i * 5)) % 30) / 100.0
                cls_id = (h >> (i * 7)) % 3
                dets.append([float(x1), float(y1), float(x2), float(y2), float(score), int(cls_id)])
            out.append(dets)
        return out
