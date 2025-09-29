# scripts/scaffold_benchmark.py
from __future__ import annotations
from pathlib import Path

def touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")

def main() -> None:
    root = Path(__file__).resolve().parents[1]  # repo root: agvision/
    # Segmentation example
    (root / "configs/benchmarks/segmentation/disease").mkdir(parents=True, exist_ok=True)
    (root / "data/benchmarks/segmentation/disease/leaf_seg_v1/images").mkdir(parents=True, exist_ok=True)
    (root / "data/benchmarks/segmentation/disease/leaf_seg_v1/masks").mkdir(parents=True, exist_ok=True)
    touch(root / "data/benchmarks/segmentation/disease/leaf_seg_v1/images/.gitkeep")
    touch(root / "data/benchmarks/segmentation/disease/leaf_seg_v1/masks/.gitkeep")
    # Classification example
    (root / "data/benchmarks/classification/leaf/leaf_cls_v1/train/healthy").mkdir(parents=True, exist_ok=True)
    (root / "data/benchmarks/classification/leaf/leaf_cls_v1/train/disease").mkdir(parents=True, exist_ok=True)
    (root / "data/benchmarks/classification/leaf/leaf_cls_v1/val/healthy").mkdir(parents=True, exist_ok=True)
    (root / "data/benchmarks/classification/leaf/leaf_cls_v1/val/disease").mkdir(parents=True, exist_ok=True)
    touch(root / "data/benchmarks/classification/leaf/leaf_cls_v1/train/healthy/.gitkeep")
    touch(root / "data/benchmarks/classification/leaf/leaf_cls_v1/train/disease/.gitkeep")
    touch(root / "data/benchmarks/classification/leaf/leaf_cls_v1/val/healthy/.gitkeep")
    touch(root / "data/benchmarks/classification/leaf/leaf_cls_v1/val/disease/.gitkeep")
    # Detection example (COCO)
    (root / "data/benchmarks/detection/fruit/fruit_det_v1/images").mkdir(parents=True, exist_ok=True)
    touch(root / "data/benchmarks/detection/fruit/fruit_det_v1/images/.gitkeep")
    print("Scaffold ready. Drop your sample images/masks into the created folders.")

if __name__ == "__main__":
    main()
