"""Segmentation and droplet level metrics."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def confusion(pred: np.ndarray, truth: np.ndarray):
    p, t = pred.astype(bool), truth.astype(bool)
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    tn = int(np.sum(~p & ~t))
    return tp, fp, fn, tn


def scores(tp, fp, fn, tn) -> Dict[str, float]:
    def div(a, b):
        return a / b if b else float("nan")
    return {
        "dice": div(2 * tp, 2 * tp + fp + fn),
        "iou": div(tp, tp + fp + fn),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "pixel_accuracy": div(tp + tn, tp + fp + fn + tn),
        "water_fraction_true": div(tp + fn, tp + fp + fn + tn),
        "water_fraction_pred": div(tp + fp, tp + fp + fn + tn),
    }


def frame_scores(pred, truth) -> Dict[str, float]:
    return scores(*confusion(pred, truth))


def pooled_scores(pairs) -> Dict[str, float]:
    """Pixel counts summed over all frames, then scores (micro average)."""
    total = np.zeros(4, dtype=np.int64)
    for pred, truth in pairs:
        total += np.array(confusion(pred, truth))
    return scores(*total)


def match_instances(pred_masks: Sequence[np.ndarray], true_masks: Sequence[np.ndarray], thr=0.5):
    """Greedy one-to-one matching by IoU. Returns (matches, iou matrix)."""
    if not len(pred_masks) or not len(true_masks):
        return [], np.zeros((len(pred_masks), len(true_masks)))
    P = np.stack([m.ravel() for m in pred_masks]).astype(np.float32)
    T = np.stack([m.ravel() for m in true_masks]).astype(np.float32)
    inter = P @ T.T
    union = P.sum(1)[:, None] + T.sum(1)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    matches, used_p, used_t = [], set(), set()
    for k in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(k, iou.shape)
        if iou[i, j] < thr:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        matches.append((int(i), int(j), float(iou[i, j])))
    return matches, iou


def instance_scores(pred_masks, true_masks, thr=0.5) -> Dict[str, float]:
    matches, _ = match_instances(pred_masks, true_masks, thr)
    tp = len(matches)
    fp = len(pred_masks) - tp
    fn = len(true_masks) - tp
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "mean_matched_iou": float(np.mean([m[2] for m in matches])) if matches else float("nan")}


def recall_by_size(pred_union: np.ndarray, true_masks: Sequence[np.ndarray], bins_px: Sequence[float],
                   min_cover=0.5):
    """For each true droplet: found if at least ``min_cover`` of it is
    predicted as water. Grouped by equivalent diameter in pixels."""
    diam = np.array([2 * np.sqrt(m.sum() / np.pi) for m in true_masks])
    found = np.array([(pred_union[m.astype(bool)].mean() if m.any() else 0) >= min_cover
                      for m in true_masks])
    out = []
    for lo, hi in zip(bins_px[:-1], bins_px[1:]):
        sel = (diam >= lo) & (diam < hi)
        out.append({"diameter_px": [float(lo), float(hi)], "droplets": int(sel.sum()),
                    "found": int(found[sel].sum()),
                    "recall": float(found[sel].mean()) if sel.any() else float("nan")})
    return out


def rasterise(polygons, shape) -> np.ndarray:
    """Label image from polygons, 1..n in the given order."""
    import cv2
    lab = np.zeros(shape, np.int32)
    for i, p in enumerate(polygons, 1):
        cv2.fillPoly(lab, [np.round(p).astype(np.int32)], int(i))
    return lab


def label_iou(pred_lab: np.ndarray, true_lab: np.ndarray) -> np.ndarray:
    """IoU between every predicted and every true region of two label images
    (0 is background). Rows: predicted ids 1..P, columns: true ids 1..T."""
    P, T = int(pred_lab.max()), int(true_lab.max())
    if not P or not T:
        return np.zeros((P, T))
    pair = pred_lab.astype(np.int64) * (T + 1) + true_lab
    inter = np.bincount(pair.ravel(), minlength=(P + 1) * (T + 1)).reshape(P + 1, T + 1)
    ap, at = inter.sum(1), inter.sum(0)
    inter = inter[1:, 1:].astype(np.float64)
    union = ap[1:, None] + at[None, 1:] - inter
    return inter / np.maximum(union, 1)


def instance_scores_labels(pred_lab, true_lab, thr=0.5) -> Dict[str, float]:
    """``instance_scores`` for label images; fast enough for full frames."""
    iou = label_iou(pred_lab, true_lab)
    P = len(np.unique(pred_lab[pred_lab > 0]))
    T = len(np.unique(true_lab[true_lab > 0]))
    matches, used_p, used_t = [], set(), set()
    if iou.size:
        for k in np.argsort(-iou, axis=None):
            i, j = np.unravel_index(k, iou.shape)
            if iou[i, j] < thr:
                break
            if i in used_p or j in used_t:
                continue
            used_p.add(i)
            used_t.add(j)
            matches.append(float(iou[i, j]))
    tp = len(matches)
    prec = tp / P if P else float("nan")
    rec = tp / T if T else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"tp": tp, "fp": P - tp, "fn": T - tp, "precision": prec, "recall": rec, "f1": f1,
            "mean_matched_iou": float(np.mean(matches)) if matches else float("nan")}
