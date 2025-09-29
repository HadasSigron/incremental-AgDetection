from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional
from pycocotools.coco import COCO
from core.datasets.base import BaseAdapter

class COCODetectionAdapter(BaseAdapter):
    """
    COCO detection adapter.

    חוזה ל-Runner:
    - len(adapter) -> מספר דגימות אחרי סינון לקבצים שקיימים פיזית
    - iter_samples() -> מחזיר מילונים אחידים: {"id": int, "path": str, "width": int?, "height": int?}
    - describe() -> לוג ידידותי ל-UI
    - image_ids() -> רשימת מזהי תמונות (לא חובה, אבל נוח למטריקות)
    """
    def __init__(self, images_dir: Path, ann_path: Path):
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.ann_path = Path(ann_path).expanduser().resolve()
        self.coco = COCO(str(self.ann_path))

        # סנן לרשומות שיש להן קובץ קיים בדיסק
        ids: List[int] = []
        for im in self.coco.loadImgs(self.coco.getImgIds()):
            p = self.images_dir / im["file_name"]
            if p.exists():
                ids.append(int(im["id"]))
        self._ids = ids

    @classmethod
    def build_from_config(cls, cfg) -> "COCODetectionAdapter":
        root = Path(str(cfg.dataset.root)).expanduser().resolve()
        images_dir = (root / (getattr(cfg.dataset, "img_dir", None) or "images")).resolve()

        ann_key = getattr(cfg.dataset, "ann_file", None) or getattr(cfg.dataset, "annotations", None)
        if not ann_key:
            raise ValueError("COCO requires dataset.ann_file (instances JSON).")

        ann_path = Path(ann_key)
        if not ann_path.is_absolute():
            ann_path = (root / ann_key).resolve()

        return cls(images_dir, ann_path)

    # ---------- API ל-Worker/Runner ----------
    def describe(self) -> Dict[str, Any]:
        return {"images": str(self.images_dir), "ann": str(self.ann_path)}

    def __len__(self) -> int:
        return len(self._ids)

    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """
        מחזיר דגימות עבור inference בפורמט אחיד:
          {"id": <int>, "path": <str>, "width": <int?>, "height": <int?>}
        """
        if not self._ids:
            return
        for img_id in self._ids:
            im = self.coco.loadImgs([img_id])[0]
            path = self.images_dir / im["file_name"]
            yield {
                "id": int(img_id),
                "path": str(path),
                "width": int(im["width"]) if "width" in im else None,
                "height": int(im["height"]) if "height" in im else None,
            }

    # לא חובה, אבל עוזר למטריקות לבחור בדיוק את התמונות שהוערכו
    def image_ids(self) -> List[int]:
        return list(self._ids)

    # שמור את __getitem__ הישן רק אם מקום אחר בקוד עוד משתמש בו (לא חובה ל-Runner הנוכחי)
    def __getitem__(self, idx: int):
        img_id = self._ids[idx]
        im = self.coco.loadImgs([img_id])[0]
        p = self.images_dir / im["file_name"]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        return p, anns, im


    def coco_gt(self):
        """Compatibility shim for metrics code: return the ground-truth COCO object."""
        return self.coco