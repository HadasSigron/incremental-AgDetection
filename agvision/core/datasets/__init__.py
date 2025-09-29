# core/datasets/__init__.py
"""
Datasets package init:
- Central place to self-register dataset adapters into the factory registry.
- Keep this file minimal and robust — any failure here prevents the app from seeing adapters.
"""

from __future__ import annotations

from core.datasets.factory import register_dataset

# --- Import adapters (they will provide build/from_config callables) ---
# NOTE: All imports are inside try/except to avoid hard-crashes if an optional adapter is missing.

# COCO detection (optional)
try:
    from core.datasets.coco_detection import COCODetectionAdapter
    register_dataset("coco_detection", COCODetectionAdapter.from_config)
    register_dataset("coco",            COCODetectionAdapter.from_config)  # alias
except Exception:
    pass

# Masks / segmentation (optional)
try:
    from core.datasets.masks_segmentation import MaskSegmentationFolderAdapter
    register_dataset("masks_segmentation", MaskSegmentationFolderAdapter.from_config)
    register_dataset("segmentation",       MaskSegmentationFolderAdapter.from_config)   # alias
    register_dataset("masks",              MaskSegmentationFolderAdapter.from_config)   # alias
except Exception:
    pass

# Classification (required for your use case)
try:
    from core.datasets.classification_folder import ClassificationFolderAdapter
    register_dataset("classification_folder", ClassificationFolderAdapter.from_config)
    register_dataset("classification",        ClassificationFolderAdapter.from_config)   # alias
    register_dataset("imagefolder",           ClassificationFolderAdapter.from_config)   # alias
except Exception:
    # If this fails, classification won't be available — surface errors in logs elsewhere.
    pass
