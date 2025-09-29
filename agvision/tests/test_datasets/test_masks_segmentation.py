import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from core.datasets.masks_segmentation import MaskSegmentationFolderAdapter

# --- helpers for tests ------------------------------------------------------

def write_rgb_image(path: Path, h: int, w: int, color: tuple = (0,0,0)):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = color
    Image.fromarray(arr).save(path)

def write_mask_png(path: Path, arr: np.ndarray):
    # arr should be HxW ints (0..K). Save as L mode PNG.
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)

# --- tests ------------------------------------------------------------------

def test_from_folders_pairs_and_size_and_iter(tmp_path: Path):
    imgs = tmp_path / "images"
    masks = tmp_path / "masks"
    imgs.mkdir()
    masks.mkdir()

    # create 3 images and corresponding masks
    for i in range(3):
        p_img = imgs / f"img_{i}.jpg"
        p_mask = masks / f"img_{i}.png"
        write_rgb_image(p_img, 8, 8, color=(i*10, i*20, i*30))
        m = np.zeros((8,8), dtype=np.uint8)
        m[0,0] = i  # create a pixel with label i
        write_mask_png(p_mask, m)

    adapter = MaskSegmentationFolderAdapter.from_folders(imgs, masks)
    assert adapter.size() == 3

    items = list(adapter)
    assert len(items) == 3
    # check structure and types
    for it in items:
        assert "id" in it and "image" in it and "mask" in it
        assert isinstance(it["id"], str)
        assert isinstance(it["image"], np.ndarray) and it["image"].ndim == 3
        assert isinstance(it["mask"], np.ndarray) and it["mask"].ndim == 2

def test_as_dataloader_batching_and_partial(tmp_path: Path):
    imgs = tmp_path / "images"
    masks = tmp_path / "masks"
    imgs.mkdir()
    masks.mkdir()

    n = 7
    for i in range(n):
        (imgs / f"img_{i}.jpg").parent.mkdir(parents=True, exist_ok=True)
        write_rgb_image(imgs / f"img_{i}.jpg", 8, 8, color=(0,0,0))
        m = np.zeros((8,8), dtype=np.uint8)
        m[0,0] = i % 2
        write_mask_png(masks / f"img_{i}.png", m)

    adapter = MaskSegmentationFolderAdapter.from_folders(imgs, masks)
    bs = 3
    batches = list(adapter.as_dataloader(batch_size=bs))
    # expected batch sizes: 3,3,1
    assert len(batches) == 3
    assert [len(b[0]) for b in batches] == [3,3,1]

    # verify contents types
    for batch_images, batch_masks, batch_meta in batches:
        assert all(isinstance(img, np.ndarray) for img in batch_images)
        assert all(isinstance(m, np.ndarray) for m in batch_masks)
        assert all(isinstance(mid, str) for mid in batch_meta)

def test_infer_num_classes_and_class_names(tmp_path: Path):
    imgs = tmp_path / "images"
    masks = tmp_path / "masks"
    imgs.mkdir()
    masks.mkdir()

    # create two masks: one uses classes 0..2, other max label 4
    write_rgb_image(imgs / "a.jpg", 4, 4)
    m1 = np.zeros((4,4), dtype=np.uint8); m1[0,0]=2
    write_mask_png(masks / "a.png", m1)

    write_rgb_image(imgs / "b.jpg", 4, 4)
    m2 = np.zeros((4,4), dtype=np.uint8); m2[0,0]=4
    write_mask_png(masks / "b.png", m2)

    adapter = MaskSegmentationFolderAdapter.from_folders(imgs, masks)
    n_classes = adapter.num_classes()
    assert n_classes == 5  # labels 0..4
    names = adapter.class_names()
    assert len(names) == n_classes
    assert names[0].startswith("class_")

def test_skip_images_without_mask_and_non_image_files(tmp_path: Path):
    imgs = tmp_path / "images"
    masks = tmp_path / "masks"
    imgs.mkdir()
    masks.mkdir()

    # create image with no mask
    write_rgb_image(imgs / "no_mask.jpg", 4, 4)
    # create a valid pair
    write_rgb_image(imgs / "ok.jpg", 4, 4)
    m = np.zeros((4,4), dtype=np.uint8)
    write_mask_png(masks / "ok.png", m)
    # create a non-image file (should be ignored)
    (imgs / "readme.txt").write_text("ignore me", encoding="utf-8")

    adapter = MaskSegmentationFolderAdapter.from_folders(imgs, masks)
    # should only include the one valid pair
    assert adapter.size() == 1
    items = list(adapter)
    assert items[0]["id"] == "ok"

def test_empty_pairs_returns_zero_size(tmp_path: Path):
    imgs = tmp_path / "images"
    masks = tmp_path / "masks"
    imgs.mkdir()
    masks.mkdir()
    adapter = MaskSegmentationFolderAdapter.from_folders(imgs, masks)
    assert adapter.size() == 0
    assert list(adapter) == []
