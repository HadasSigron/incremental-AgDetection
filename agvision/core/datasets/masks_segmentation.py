# core/datasets/masks_segmentation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any

from core.datasets.base import BaseAdapter  # << חדש: חוזה אדאפטרים

# Supported image extensions (case-insensitive on Windows; we still normalize)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_MASK_EXT = ".png"  # masks are typically PNGs


@dataclass
class _SegSample:
    """Internal normalized sample representation."""
    id: int
    path: str           # image path
    mask_path: str      # mask path (PNG)
    label: int = 0      # not used for semantic seg; kept for uniformity


class MaskSegmentationFolderAdapter(BaseAdapter):  # << חדש: יורש מ-BaseAdapter
    """
    A robust folder-based segmentation adapter.

    Features:
    - Recursively scans `images_dir` for images with supported extensions.
    - For each image, looks for a mask with the same stem (file name without extension)
      anywhere under `masks_dir`, using a recursive search.
    - Skips images without a matching mask (with a helpful counter).
    - Exposes `iter_samples()` yielding normalized dicts:
        {"id": int, "path": str, "mask_path": str, "label": int}
    - Supports `len(adapter)`.

    This keeps the runner contract simple and decoupled from directory layout.
    """

    # -------- חדש: בנייה מקונפיג (Plugin-style) --------
    @classmethod
    def build_from_config(cls, cfg) -> "MaskSegmentationFolderAdapter":
        """
        Build adapter directly from AppConfig-style cfg:
          cfg.dataset.root, img_dir, mask_dir
        """
        root = Path(str(cfg.dataset.root)).expanduser().resolve()
        images_dir = (root / (getattr(cfg.dataset, "img_dir", None) or "images")).resolve()
        masks_dir  = (root / (getattr(cfg.dataset, "mask_dir", None) or "masks")).resolve()
        return cls(images_dir, masks_dir)

    # -------- חדש: תיאור סטנדרטי ללוג של ה-Worker --------
    def describe(self) -> Dict[str, Any]:
        return {"images": str(self.images_dir), "masks": str(self.masks_dir)}

    def __init__(self, images_dir: Path, masks_dir: Path) -> None:
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.masks_dir = Path(masks_dir).expanduser().resolve()
        self._samples: List[_SegSample] = []
        self._build_index()

    @classmethod
    def from_folders(cls, images_dir: str, masks_dir: str) -> "MaskSegmentationFolderAdapter":
        return cls(Path(images_dir).expanduser().resolve(),
                   Path(masks_dir).expanduser().resolve())

    # ------------------------ public API ------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        for s in self._samples:
            yield {
                "id": s.id,
                "path": s.path,
                "mask_path": s.mask_path,
                "label": s.label,
            }

    # ------------------------ internals ------------------------

    @staticmethod
    def _is_image(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in _IMAGE_EXTS

    def _find_mask_for(self, stem: str) -> Optional[Path]:
        """
        Search for a mask file named '<stem>.png' anywhere under masks_dir.
        If multiple found, pick the first deterministic (sorted) one.
        """
        cand = sorted(self.masks_dir.rglob(stem + _MASK_EXT))
        return cand[0] if cand else None

    def _build_index(self) -> None:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks folder not found: {self.masks_dir}")

        imgs = [p for p in self.images_dir.rglob("*") if self._is_image(p)]
        if not imgs:
            # Don't raise; let caller print dataset size and hint about extensions/layout.
            self._samples = []
            return

        samples: List[_SegSample] = []
        skipped = 0
        for idx, img_path in enumerate(sorted(imgs)):
            m = self._find_mask_for(img_path.stem)
            if not m:
                skipped += 1
                continue
            samples.append(_SegSample(id=idx, path=str(img_path), mask_path=str(m)))

        self._samples = samples
        # Optionally log counts via print; UI worker already shows sizes, so we stay quiet here.
        # print(f"[MaskSeg] indexed: {len(samples)} samples (skipped:{skipped} without masks)")
