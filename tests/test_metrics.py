import numpy as np

from dropseg.metrics import (frame_scores, instance_scores, instance_scores_labels, label_iou,
                             pooled_scores, rasterise)


def test_frame_scores_perfect_and_empty():
    m = np.zeros((10, 10), np.uint8)
    m[2:5, 2:5] = 1
    s = frame_scores(m, m)
    assert s["dice"] == 1.0 and s["iou"] == 1.0
    s = frame_scores(np.zeros_like(m), m)
    assert s["dice"] == 0.0 and s["recall"] == 0.0


def test_pooled_is_micro_average():
    a = np.zeros((4, 4), np.uint8)
    a[:2] = 1
    b = np.zeros((4, 4), np.uint8)
    b[0] = 1
    s = pooled_scores([(a, a), (a, b)])
    # tp 8 + 4, fp 0 + 4, fn 0
    assert abs(s["dice"] - 24 / 28) < 1e-9


def test_label_iou_and_matching():
    t = np.zeros((20, 20), np.int32)
    t[2:6, 2:6] = 1
    t[10:14, 10:14] = 2
    p = np.zeros_like(t)
    p[2:6, 2:6] = 1            # exact
    p[10:14, 12:16] = 2        # half overlap, iou 1/3
    p[0:2, 18:20] = 3          # false positive
    iou = label_iou(p, t)
    assert iou.shape == (3, 2)
    assert iou[0, 0] == 1.0 and abs(iou[1, 1] - 1 / 3) < 1e-9
    s = instance_scores_labels(p, t, thr=0.5)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 2, 1)
    s2 = instance_scores([p == i for i in (1, 2, 3)], [t == i for i in (1, 2)], thr=0.5)
    assert (s2["tp"], s2["fp"], s2["fn"]) == (1, 2, 1)


def test_rasterise_polygons():
    sq = np.array([[2, 2], [7, 2], [7, 7], [2, 7]], np.float32)
    lab = rasterise([sq, sq + 10], (20, 20))
    assert set(np.unique(lab)) == {0, 1, 2}
    assert (lab == 1).sum() == (lab == 2).sum() == 36
