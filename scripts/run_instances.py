#!/usr/bin/env python3
"""Single droplets: Mask R-CNN against semantic segmentation + watershed.

    python scripts/run_instances.py [--iters 400]

Only five frames of recording a have every droplet outlined (a20, a25, a30,
a35, a65). Mask R-CNN is trained on full resolution crops in three folds;
the frames next to a test frame (2.5 min) are left out of training:

    fold 0: test a20, a25   train a35, a65
    fold 1: test a30, a35   train a20, a65
    fold 2: test a65        train a20, a25, a30, a35

The semantic models are scored on the same frames, using their held-out
probability maps from scripts/run_cv.py, split into droplets with a
watershed. Watershed on the ground truth mask is given as well: it shows how
much is lost by the splitting step alone.

Writes results/instances/summary.json and the predicted label images.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from dropseg.data import load_frame, read_index  # noqa: E402
from dropseg.droplets import split_droplets  # noqa: E402
from dropseg.metrics import frame_scores, instance_scores_labels, rasterise  # noqa: E402

OUT = ROOT / "results" / "instances"
CV = ROOT / "results" / "cv"
FOLDS = [(["A_a035", "A_a065"], ["A_a020", "A_a025"]),
         (["A_a020", "A_a065"], ["A_a030", "A_a035"]),
         (["A_a020", "A_a025", "A_a030", "A_a035"], ["A_a065"])]
SIZE_BINS = [0, 10, 20, 40, 80, 10000]          # equivalent diameter, full frame pixels
SPLIT = dict(min_area_px=12, min_peak_distance=10)  # watershed settings at full resolution


def recall_by_size(pred_water, true_lab):
    ids = np.unique(true_lab[true_lab > 0])
    area = np.bincount(true_lab.ravel(), minlength=ids.max() + 1)[ids]
    wet = np.bincount(true_lab.ravel(), weights=pred_water.ravel().astype(float),
                      minlength=ids.max() + 1)[ids]
    diam = 2 * np.sqrt(area / np.pi)
    found = wet / np.maximum(area, 1) >= 0.5
    out = []
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        sel = (diam >= lo) & (diam < hi)
        out.append({"diameter_px": [lo, hi], "droplets": int(sel.sum()), "found": int(found[sel].sum())})
    return out


def run_maskrcnn(frames, iters):
    from dropseg.models import maskrcnn
    (OUT / "maskrcnn").mkdir(parents=True, exist_ok=True)
    for k, (tr, te) in enumerate(FOLDS):
        done = OUT / "maskrcnn" / ("fold%d.json" % k)
        if done.is_file():
            continue
        t0 = time.time()
        cfg = maskrcnn.MRCNNConfig(iters=iters, seed=k, threads=8)
        model, hist = maskrcnn.train([frames[i] for i in tr], cfg, log=lambda *a: None)
        fit_s = time.time() - t0
        times = {}
        for fid in te:
            t1 = time.time()
            lab, _ = maskrcnn.predict(model, frames[fid].image)
            times[fid] = round(time.time() - t1, 1)
            cv2.imwrite(str(OUT / "maskrcnn" / ("%s.png" % fid)), lab.astype(np.uint16))
        done.write_text(json.dumps({"fold": k, "train": tr, "test": te, "fit_s": round(fit_s),
                                    "predict_s": times, "history": hist}, indent=1))
        print("maskrcnn fold %d done (%.0f min)" % (k, fit_s / 60), flush=True)


def semantic_labels(method, fid, shape):
    p = cv2.imread(str(CV / method / "pred" / ("session_A__%s.png" % fid)), 0)
    if p is None:
        return None
    p = cv2.resize(p, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) >= 128
    return split_droplets(p, **SPLIT)


def evaluate(frames):
    rows = []
    for fid, f in frames.items():
        true_lab = rasterise(f.instances, f.mask.shape)
        preds = {"watershed on true mask": split_droplets(true_lab > 0, **SPLIT)}
        mr = OUT / "maskrcnn" / ("%s.png" % fid)
        if mr.is_file():
            preds["mask r-cnn"] = cv2.imread(str(mr), cv2.IMREAD_UNCHANGED).astype(np.int32)
        for m, name in [("resnet18", "u-net resnet18 + watershed"), ("plain", "u-net plain + watershed"),
                        ("forest", "pixel forest + watershed")]:
            lab = semantic_labels(m, fid, f.mask.shape)
            if lab is not None:
                preds[name] = lab
        for name, lab in preds.items():
            s = instance_scores_labels(lab, true_lab)
            s.update({"frame": fid, "method": name, "true_count": int(len(f.instances)),
                      "pred_count": int(len(np.unique(lab[lab > 0]))),
                      "dice": frame_scores(lab > 0, true_lab > 0)["dice"],
                      "by_size": recall_by_size(lab > 0, true_lab)})
            rows.append(s)
    return rows


def summarise(rows):
    out = {}
    for name in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == name]
        tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
        p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        sizes = []
        for b in range(len(SIZE_BINS) - 1):
            n = sum(r["by_size"][b]["droplets"] for r in rs)
            found = sum(r["by_size"][b]["found"] for r in rs)
            sizes.append({"diameter_px": SIZE_BINS[b:b + 2], "droplets": n, "found": found,
                          "recall": found / n if n else None})
        out[name] = {"frames": len(rs), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": rc,
                     "f1": 2 * p * rc / (p + rc) if tp else 0.0,
                     "count_error_mean": float(np.mean([abs(r["pred_count"] - r["true_count"]) / r["true_count"]
                                                        for r in rs])),
                     "dice_mean": float(np.mean([r["dice"] for r in rs])), "recall_by_size": sizes}
        print("%-28s P %.3f R %.3f F1 %.3f  count error %.0f%%  dice %.3f" % (
            name, p, rc, out[name]["f1"], 100 * out[name]["count_error_mean"], out[name]["dice_mean"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    rows = {r["id"]: r for r in read_index()}
    ids = sorted({i for tr, te in FOLDS for i in tr + te})
    frames = {i: load_frame(rows[i], scale=1.0) for i in ids}
    if not args.skip_train:
        run_maskrcnn(frames, args.iters)
    per_frame = evaluate(frames)
    summary = summarise(per_frame)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "frames": per_frame}, indent=1))


if __name__ == "__main__":
    main()
