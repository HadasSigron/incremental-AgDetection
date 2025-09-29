"""TorchScript classifier wrapper (placeholder)."""

from __future__ import annotations
from typing import List, Tuple, Optional
from pathlib import Path
import hashlib

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class TorchscriptClassifier:
    """
    Loads a TorchScript classification model:
    - Expects model(path).forward(tensor[N,3,H,W]) -> logits[N,C]
    - Applies ImageNet-like preprocessing by default
    """
    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 224):
        p = Path(weights)
        if not p.exists():
            raise FileNotFoundError(f"model weights not found: {weights}")
        try:
            self.model = torch.jit.load(str(p), map_location=device)
        except Exception:
            # fallback for saved state_dict scripted with torch.save
            self.model = torch.load(str(p), map_location=device)
        self.model.eval()
        self.device = torch.device(device)
        self.imgsz = int(imgsz)

        self.tf = transforms.Compose([
            transforms.Resize(self.imgsz),
            transforms.CenterCrop(self.imgsz),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict_batch(self, paths: List[str]) -> Tuple[List[int], Optional[List[List[float]]]]:
        batch = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            batch.append(self.tf(img))
        x = torch.stack(batch, dim=0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1).tolist()
        return preds, probs.tolist()


class SimulatedClassifier:
    """Deterministic pseudo-classifier for tests/dev: hash(path) % n_classes."""
    def __init__(self, n_classes: int = 2):
        self.n = int(n_classes)

    def predict_batch(self, paths: List[str]) -> Tuple[List[int], Optional[List[List[float]]]]:
        preds: List[int] = []
        probs: List[List[float]] = []
        for p in paths:
            h = int(hashlib.sha1(str(p).encode("utf-8")).hexdigest(), 16)
            c = h % max(1, self.n)
            preds.append(int(c))
            vec = [0.0] * self.n
            if self.n > 0:
                vec[c] = 1.0
            probs.append(vec)
        return preds, probs
