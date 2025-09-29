from __future__ import annotations
import numpy as np
import math
import pytest
from core.metrics.segmentation import compute_per_class_iou_and_dice_batch

def assert_close(a, b, tol=1e-8):
    assert math.isclose(a, b, rel_tol=1e-6, abs_tol=tol)

def test_perfect_match_single_image():
    gt = np.array([[[0,1],[1,0]]], dtype=np.uint8)   # shape (1,2,2)
    pred = gt.copy()
    res = compute_per_class_iou_and_dice_batch(gt, pred)
    # classes are 0 and 1
    assert_close(res["per_class_iou"][0], 1.0)
    assert_close(res["per_class_iou"][1], 1.0)
    assert_close(res["per_class_dice"][0], 1.0)
    assert_close(res["per_class_dice"][1], 1.0)
    assert_close(res["mean_iou"], 1.0)
    assert_close(res["mean_dice"], 1.0)

def test_partial_known_values_small():
    # GT:
    # [[0,1],
    #  [1,0]]
    # P:
    # [[0,1],
    #  [0,0]]
    gt = np.array([[[0,1],[1,0]]], dtype=np.uint8)
    pred = np.array([[[0,1],[0,0]]], dtype=np.uint8)
    res = compute_per_class_iou_and_dice_batch(gt, pred, n_classes=2)
    # class0: intersection = 2, union = 3 -> IoU=2/3 ; Dice = 2*2/(3+2)=4/5 = 0.8
    # class1: intersection = 1, union = 2 -> IoU=1/2 ; Dice = 2*1/(1+1)=1.0? careful
    # Wait: for class1: gt_sum=2, pred_sum=1 -> Dice = 2*1/(2+1)=2/3
    assert_close(res["per_class_iou"][0], 2.0/3.0)
    assert_close(res["per_class_dice"][0], 0.8)
    assert_close(res["per_class_iou"][1], 0.5)
    assert_close(res["per_class_dice"][1], 2.0/3.0)

def test_empty_class_both_empty():
    # both GT and pred have no pixels of class 2 -> IoU/Dice == 1.0 by convention
    gt = np.zeros((2,4,4), dtype=np.uint8)          # all zeros
    pred = np.zeros((2,4,4), dtype=np.uint8)
    # add one pixel of class 1 in both so class 0 and 1 exist, but class 2 absent
    gt[0,0,0] = 1
    pred[0,0,0] = 1
    res = compute_per_class_iou_and_dice_batch(gt, pred, n_classes=3)
    assert_close(res["per_class_iou"][2], 1.0)
    assert_close(res["per_class_dice"][2], 1.0)

def test_class_in_gt_not_in_pred():
    gt = np.zeros((1,3,3), dtype=np.uint8)
    pred = np.zeros((1,3,3), dtype=np.uint8)
    # set some pixels of class 1 in GT, none in pred
    gt[0, 0:2, 0:2] = 1
    res = compute_per_class_iou_and_dice_batch(gt, pred, n_classes=2)
    assert_close(res["per_class_iou"][1], 0.0)
    assert_close(res["per_class_dice"][1], 0.0)

def test_n_classes_overrides():
    gt = np.zeros((1,2,2), dtype=np.uint8)
    pred = np.zeros((1,2,2), dtype=np.uint8)
    # both only class 0 present, but n_classes forces 4 classes
    res = compute_per_class_iou_and_dice_batch(gt, pred, n_classes=4)
    # per_class_iou should have 4 entries
    assert len(res["per_class_iou"]) == 4
    assert len(res["per_class_dice"]) == 4

def test_shape_mismatch_raises():
    gt = np.zeros((2,4,4), dtype=np.uint8)
    pred = np.zeros((3,4,4), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_per_class_iou_and_dice_batch(gt, pred)

def test_numpy_types_and_scalars():
    # use numpy float/int types inside arrays and as potential scalars
    gt = np.array([[[0,1],[1,0]]], dtype=np.int32)
    pred = np.array([[[0,1],[1,0]]], dtype=np.int64)
    res = compute_per_class_iou_and_dice_batch(gt, pred)
    assert_close(res["mean_iou"], 1.0)
    assert_close(res["mean_dice"], 1.0)
