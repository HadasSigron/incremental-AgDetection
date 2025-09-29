# agvision/core/runners/segmenter_runner.py
from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from PIL import Image

from core.types.runner_result import ensure_runner_result  # convert & validate

logger = logging.getLogger(__name__)

# ========================= helpers: io & viz =========================

def ensure_folder(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_mask_png(mask: np.ndarray, path: str):
    Image.fromarray(mask.astype(np.uint8)).save(path)

def make_overlay(image: np.ndarray, mask: np.ndarray, alpha=0.5):
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    img = Image.fromarray(image.astype(np.uint8)).convert("RGBA")
    h, w = mask.shape[:2]
    palette = [
        (0,0,0,0),
        (255,0,0,int(255*alpha)),
        (0,255,0,int(255*alpha)),
        (0,0,255,int(255*alpha)),
        (255,255,0,int(255*alpha)),
    ]
    overlay = Image.new("RGBA", (w,h), (0,0,0,0))
    mask_uint8 = mask.astype(np.uint8)
    for cls in np.unique(mask_uint8):
        if int(cls) == 0:
            continue
        color = palette[int(cls) % len(palette)]
        cls_mask = Image.fromarray((mask_uint8 == cls).astype(np.uint8) * 255)
        color_layer = Image.new("RGBA", (w,h), color)
        overlay.paste(color_layer, (0,0), cls_mask)
    blended = Image.alpha_composite(img, overlay)
    return blended.convert("RGB")

def compute_per_class_iou_and_dice_batch(gt_masks, pred_masks, n_classes: Optional[int]=None):
    assert gt_masks.shape == pred_masks.shape
    if n_classes is None:
        n_classes = int(max(int(gt_masks.max()), int(pred_masks.max())) + 1)
    eps = 1e-7
    per_class_iou = {}
    per_class_dice = {}
    for c in range(n_classes):
        gt_c = (gt_masks == c)
        pred_c = (pred_masks == c)
        inter = int((gt_c & pred_c).sum())
        union = int((gt_c | pred_c).sum())
        gt_sum = int(gt_c.sum())
        pred_sum = int(pred_c.sum())
        iou = 1.0 if union == 0 else inter / (union + eps)
        dice = 1.0 if (gt_sum + pred_sum) == 0 else (2.0 * inter) / (gt_sum + pred_sum + eps)
        per_class_iou[c] = float(iou)
        per_class_dice[c] = float(dice)
    mean_iou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0
    mean_dice = float(np.mean(list(per_class_dice.values()))) if per_class_dice else 0.0
    return {"per_class_iou": per_class_iou, "per_class_dice": per_class_dice, "mean_iou": mean_iou, "mean_dice": mean_dice}

def is_cancelled(cancel_token) -> bool:
    if cancel_token is None: return False
    if hasattr(cancel_token, "is_set") and callable(cancel_token.is_set):
        return cancel_token.is_set()
    if hasattr(cancel_token, "cancelled"):
        return bool(getattr(cancel_token, "cancelled"))
    if callable(cancel_token):
        return bool(cancel_token())
    return False

# ========================= internal producers =========================
# Their sole goal: pull eval-only/model-logic out of the runner (no extra files).

class _SegPredictionsProducer:
    """Produces segmentation outputs and artifacts."""
    def produce(self, dataset, config: Dict[str, Any], cancel_token) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

class _SegFromModelProducer(_SegPredictionsProducer):
    """Run a model wrapper that exposes predict(batch_images) -> (B,H,W)."""

    def __init__(self, model):
        self.model = model

    def _build_dataloader(self, dataset, batch_size: int):
        if hasattr(dataset, "as_dataloader"):
            return dataset.as_dataloader(batch_size=batch_size)
        # fallback: simple generator
        def gen():
            bimgs, bmasks, bmeta = [], [], []
            for item in dataset:
                bimgs.append(item["image"])
                bmasks.append(item["mask"])
                bmeta.append(item.get("id", None))
                if len(bimgs) >= batch_size:
                    yield bimgs, bmasks, bmeta
                    bimgs, bmasks, bmeta = [], [], []
            if bimgs:
                yield bimgs, bmasks, bmeta
        return gen()

    def produce(self, dataset, config: Dict[str, Any], cancel_token):
        t0 = time.time()

        # honor explicit out_dir; otherwise use output_root/timestamp
        out_dir = str(config.get("out_dir", "")).strip()
        output_root = str(config.get("output_root", "runs/segmentation"))
        if out_dir:
            run_dir = out_dir
        else:
            ts = time.strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(output_root, ts)

        preds_dir = os.path.join(run_dir, "predicted_masks")
        overlays_dir = os.path.join(run_dir, "overlays")
        save_artifacts = bool(config.get("save_artifacts", True)) and not bool(config.get("no_write", False))
        if save_artifacts:
            ensure_folder(preds_dir); ensure_folder(overlays_dir)

        batch_size = int(config.get("batch_size", 4))
        n_classes = config.get("n_classes", None)

        dataloader = self._build_dataloader(dataset, batch_size)

        all_gt, all_pred = [], []
        processed = 0

        for batch in dataloader:
            if is_cancelled(cancel_token):
                logger.info("SegmenterRunner cancelled after %d samples", processed)
                break

            if isinstance(batch, tuple) and len(batch) == 3:
                batch_images, batch_gt, batch_meta = batch
            else:
                items = batch
                batch_images = [it["image"] for it in items]
                batch_gt = [it["mask"] for it in items]
                batch_meta = [it.get("id", i) for i, it in enumerate(items)]

            # normalize to uint8 3-ch
            imgs_np = np.stack([
                (img.astype(np.uint8) if img.ndim == 3 else np.repeat(img[:, :, None], 3, axis=2).astype(np.uint8))
                for img in batch_images
            ], axis=0)

            preds = self.model.predict(imgs_np)  # (B,H,W) int masks
            preds = np.asarray(preds)
            gts = np.stack([m.astype(np.uint8) for m in batch_gt], axis=0)

            all_gt.append(gts); all_pred.append(preds)

            if save_artifacts:
                for i, meta in enumerate(batch_meta):
                    sid = meta if isinstance(meta, (str,int)) else f"sample_{processed + i}"
                    save_mask_png(preds[i], os.path.join(preds_dir, f"{sid}.png"))
                    ov = make_overlay(imgs_np[i], preds[i])
                    ov.save(os.path.join(overlays_dir, f"{sid}.jpg"))

            processed += preds.shape[0]

        if len(all_gt) == 0:
            # metrics must be numeric-only; human messages go to artifacts
            duration = time.time() - t0
            metrics = {
                "n_samples": 0,
                "mean_iou": 0.0,
                "mean_dice": 0.0,
                "duration_sec": float(duration),
            }
            artifacts = {
                "run_dir": (run_dir if save_artifacts else None),
                "predicted_masks_dir": (preds_dir if save_artifacts else None),
                "overlays_dir": (overlays_dir if save_artifacts else None),
                "error": "no samples processed",
            }
            return metrics, artifacts

        gt_arr = np.concatenate(all_gt, axis=0)
        pred_arr = np.concatenate(all_pred, axis=0)

        # prefer core.metrics.segmentation if available
        try:
            from agvision.core.metrics import segmentation as seg_metrics  # type: ignore
            compute_fn = getattr(seg_metrics, "compute_segmentation_metrics", None) or getattr(seg_metrics, "evaluate", None)
            if compute_fn:
                metric_res = compute_fn(gt_arr, pred_arr, n_classes=n_classes)
            else:
                metric_res = compute_per_class_iou_and_dice_batch(gt_arr, pred_arr, n_classes=n_classes)
        except Exception:
            metric_res = compute_per_class_iou_and_dice_batch(gt_arr, pred_arr, n_classes=n_classes)

        duration = time.time() - t0
        metrics = {
            "n_samples": int(gt_arr.shape[0]),
            "duration_sec": float(duration),
            **metric_res,
        }
        artifacts = {
            "run_dir": (run_dir if save_artifacts else None),
            "predicted_masks_dir": (preds_dir if save_artifacts else None),
            "overlays_dir": (overlays_dir if save_artifacts else None),
        }
        return metrics, artifacts

class _SegFromDirProducer(_SegPredictionsProducer):
    """
    Eval-only mode: use existing predicted masks on disk instead of a model.
    Expect config['pred_masks_dir'] with PNGs named by sample id (e.g., <id>.png).
    """
    def produce(self, dataset, config: Dict[str, Any], cancel_token):
        pred_dir = str(config["pred_masks_dir"]).strip()
        if not pred_dir or not os.path.isdir(pred_dir):
            metrics = {"n_samples": 0, "mean_iou": 0.0, "mean_dice": 0.0}
            artifacts = {"run_dir": None, "error": "pred_masks_dir missing"}
            return metrics, artifacts

        output_root = str(config.get("output_root", "runs/segmentation_evalonly"))
        save_artifacts = bool(config.get("save_artifacts", True)) and not bool(config.get("no_write", False))

        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(output_root, ts)
        overlays_dir = os.path.join(run_dir, "overlays")
        if save_artifacts:
            ensure_folder(overlays_dir)

        all_gt, all_pred = [], []
        processed = 0

        # minimal iterator: dataset expected to yield dicts with keys: image, mask, id
        iterator = dataset.as_dataloader(batch_size=1) if hasattr(dataset, "as_dataloader") else dataset
        for item in iterator:
            if is_cancelled(cancel_token):
                break
            if isinstance(item, tuple) and len(item) == 3:
                imgs, gts, metas = item
                img = imgs[0]; gt = gts[0]; sid = metas[0]
            else:
                img = item["image"]; gt = item["mask"]; sid = item.get("id", processed)

            pred_path = os.path.join(pred_dir, f"{sid}.png")
            if not os.path.exists(pred_path):
                # skip missing pred; keep shapes aligned by continuing without append
                continue

            pred = np.array(Image.open(pred_path)).astype(np.uint8)
            all_gt.append(gt.astype(np.uint8)[None, ...])
            all_pred.append(pred[None, ...])

            if save_artifacts:
                ov = make_overlay(img, pred)
                ov.save(os.path.join(overlays_dir, f"{sid}.jpg"))

            processed += 1

        if len(all_gt) == 0:
            metrics = {"n_samples": 0, "mean_iou": 0.0, "mean_dice": 0.0}
            artifacts = {"run_dir": (run_dir if save_artifacts else None), "error": "no matched predictions"}
            return metrics, artifacts

        gt_arr = np.concatenate(all_gt, axis=0)
        pred_arr = np.concatenate(all_pred, axis=0)

        try:
            from agvision.core.metrics import segmentation as seg_metrics  # type: ignore
            compute_fn = getattr(seg_metrics, "compute_segmentation_metrics", None) or getattr(seg_metrics, "evaluate", None)
            if compute_fn:
                metric_res = compute_fn(gt_arr, pred_arr, n_classes=config.get("n_classes"))
            else:
                metric_res = compute_per_class_iou_and_dice_batch(gt_arr, pred_arr, n_classes=config.get("n_classes"))
        except Exception:
            metric_res = compute_per_class_iou_and_dice_batch(gt_arr, pred_arr, n_classes=config.get("n_classes"))

        metrics = {"n_samples": int(gt_arr.shape[0]), **metric_res}
        artifacts = {"run_dir": (run_dir if save_artifacts else None), "overlays_dir": (overlays_dir if save_artifacts else None)}
        return metrics, artifacts

# ========================= small loader facade =========================

def _load_segmenter_from_config(config: Dict[str, Any]):
    """
    Try to load a segmenter via Model Registry if available; otherwise
    accept a user-provided wrapper in config['model'] that exposes .predict(batch)->(B,H,W).
    """
    # Try model registry if present
    try:
        import agvision.core.models as _models  # noqa: F401
        from agvision.core.models import registry as _reg  # type: ignore
        loader = getattr(_reg, "load_segmenter", None)
        if callable(loader):
            return loader(config)
    except Exception:
        pass

    # Fallback: accept provided wrapper with .predict
    if "model" in config and hasattr(config["model"], "predict"):
        return config["model"]

    # As a last resort, allow a factory function in config
    if callable(config.get("model_factory")):
        return config["model_factory"](config)

    raise RuntimeError("Please provide a segmenter model wrapper (config['model']) "
                       "or implement models.registry.load_segmenter(config).")

def _build_seg_producer(config: Dict[str, Any]) -> _SegPredictionsProducer:
    if config.get("pred_masks_dir"):
        return _SegFromDirProducer()
    model = _load_segmenter_from_config(config)
    return _SegFromModelProducer(model)

# ========================= public runner =========================

class SegmenterRunner:
    """
    Segmentation runner (no ifs inside):
    - Inline producers handle 'eval-only' (pred_masks_dir) or model inference.
    - Model loading prefers Model Registry if it exposes load_segmenter(config),
      otherwise uses config['model'] wrapper that implements .predict(batch)->(B,H,W).
    """
    def __init__(self):
        pass

    def run(self, dataset, config: Dict[str, Any], cancel_token=None):
        producer = _build_seg_producer(config)
        metrics, artifacts = producer.produce(dataset, config, cancel_token)
        # Normalize to RunnerResult dict
        return ensure_runner_result({"metrics": metrics, "artifacts": artifacts})
