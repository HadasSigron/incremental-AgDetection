# scripts/convert_yolo_seg_to_masks.py
from __future__ import annotations

"""
Convert YOLO segmentation label files (polygon .txt with normalized coords)
into per-pixel class-index PNG masks that match each image 1:1.

Output layout (created if missing):
  data/benchmarks/segmentation/coco8-seg-masks/
    images/train, images/val     # copies of source images
    masks/train,  masks/val      # generated indexed PNG masks
    classes.txt                  # background + class_<id> per encountered class
Also writes a ready-to-use YAML under:
  configs/benchmarks/segmentation/demo/coco8_seg_masks.yaml
"""

from pathlib import Path
from typing import List, Tuple, Dict, Set
import shutil

import numpy as np
from PIL import Image, ImageDraw
import yaml


def _read_yolo_poly_file(fp: Path) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    Read a YOLO-seg .txt file.
    Each line: <class_id> x1 y1 x2 y2 ... (normalized [0..1] coords, pairs form a polygon)
    Returns a list of (class_id, [(x, y), ...]) with normalized floats.
    """
    if not fp.exists():
        return []
    polys = []
    for line in fp.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0 or len(coords) < 6:
            # invalid polygon (need at least 3 points)
            continue
        pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        polys.append((cls, pts))
    return polys


def _rasterize_mask(img_w: int, img_h: int, polys: List[Tuple[int, List[Tuple[float, float]]]]) -> Image.Image:
    """
    Create an index mask (mode='L'): 0=background, k>=1 => class index +1.
    We offset classes by +1 so background stays 0.
    """
    mask = Image.new("L", (img_w, img_h), color=0)
    draw = ImageDraw.Draw(mask)
    for cls, norm_pts in polys:
        # scale normalized coords to absolute pixels
        abs_pts = [(int(x * img_w), int(y * img_h)) for (x, y) in norm_pts]
        # fill polygon with class index + 1
        draw.polygon(abs_pts, fill=int(cls) + 1)
    return mask


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def convert_coco8_seg(repo_root: Path) -> Dict[str, any]:
    """
    Convert YOLO-seg dataset under:
      data/benchmarks/coco8-seg/{images,labels}/{train,val}
    into:
      data/benchmarks/segmentation/coco8-seg-masks/{images,masks}/{train,val}
    Also writes classes.txt and a ready YAML config for the UI.
    """
    src_root = repo_root / "agvision" / "data" / "benchmarks" / "coco8-seg"
    images_root = src_root / "images"
    labels_root = src_root / "labels"

    if not images_root.exists() or not labels_root.exists():
        raise SystemExit(f"Source YOLO-seg folders not found under: {src_root}")

    out_root = repo_root / "agvision" / "data" / "benchmarks" / "segmentation" / "coco8-seg-masks"
    out_images = {s: out_root / "images" / s for s in ("train", "val")}
    out_masks  = {s: out_root / "masks"  / s for s in ("train", "val")}
    for p in [*out_images.values(), *out_masks.values()]:
        p.mkdir(parents=True, exist_ok=True)

    seen_classes: Set[int] = set()
    stats = {"converted": 0, "missing_labels": 0}

    for split in ("train", "val"):
        src_img_dir = images_root / split
        src_lbl_dir = labels_root / split
        for img_fp in sorted(src_img_dir.glob("*.*")):
            if img_fp.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            lbl_fp = src_lbl_dir / (img_fp.stem + ".txt")
            polys = _read_yolo_poly_file(lbl_fp)

            # copy image
            _copy_image(img_fp, out_images[split] / img_fp.name)

            # open to get size
            with Image.open(img_fp) as im:
                w, h = im.size

            if polys:
                for cls, _ in polys:
                    seen_classes.add(int(cls))
                mask = _rasterize_mask(w, h, polys)
            else:
                # no labels -> background-only mask
                mask = Image.new("L", (w, h), color=0)
                stats["missing_labels"] += 1

            mask.save(out_masks[split] / f"{img_fp.stem}.png")
            stats["converted"] += 1

    # classes.txt: background + class_<id+1> (since mask uses +1 offset)
    classes_txt = out_root / "classes.txt"
    classes = ["background"] + [f"class_{cid+1}" for cid in sorted(seen_classes)]
    classes_txt.write_text("\n".join(classes), encoding="utf-8")

    # write ready YAML
    yaml_fp = repo_root / "agvision" / "configs" / "benchmarks" / "segmentation" / "demo" / "coco8_seg_masks.yaml"
    yaml_fp.parent.mkdir(parents=True, exist_ok=True)
    yaml_data = {
        "task": "Segmentation",
        "dataset": {
            "kind": "masks_segmentation",
            "img_dir": "data/benchmarks/segmentation/coco8-seg-masks/images",
            "mask_dir": "data/benchmarks/segmentation/coco8-seg-masks/masks",
            "classes": classes,
            "transforms": {"resize": [640, 640]},
        },
        "model": {
            "format": "onnx",
            "path": "models/weights/yolov8n-seg.onnx",
            "input_size": [640, 640],
            "conf_threshold": 0.25,
            "iou_threshold": 0.5,
        },
        "eval": {
            "batch_size": 2,
            "num_workers": 2,
            "device": "cpu",
            "metrics": ["miou", "pixel_accuracy"],
        },
        "paths": {"runs_root": "runs"},
    }
    yaml_fp.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    return {
        "out_root": str(out_root),
        "yaml": str(yaml_fp),
        "classes": classes,
        "stats": stats,
    }


if __name__ == "__main__":
    # Run from repo root parent (folder that contains "agvision/")
    repo_root = Path(__file__).resolve().parents[2]  # .../<repo_root>
    info = convert_coco8_seg(repo_root)
    print("Output root:", info["out_root"])
    print("YAML:", info["yaml"])
    print("Classes:", info["classes"])
    print("Stats:", info["stats"])
