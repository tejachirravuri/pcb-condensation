#!/usr/bin/env python3
"""Cross-validation of the segmentation methods.

    python scripts/run_cv.py --methods forest plain resnet18 [--iters 800]

1. Session A (recording a, 14 frames): 5 folds of consecutive frames, frames
   within 2.5 min of the test block removed from training.
2. Other camera setup (session R, 6 frames): trained on all of session A,
   tested on R; and A plus half of R tested on the other half.

Each fold writes results/cv/<method>/<experiment>_fold<k>.json and the
held-out probability maps to results/cv/<method>/pred/<frame>.png, so an
interrupted run continues where it stopped.
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

from dropseg.data import load_frames, time_block_folds  # noqa: E402
from dropseg.metrics import frame_scores, pooled_scores  # noqa: E402

OUT = ROOT / "results" / "cv"


def fit_predict(method, train_frames, test_frames, iters, seed=0):
    t0 = time.time()
    if method == "forest":
        from dropseg.baseline import PixelForest
        model = PixelForest(seed=seed).fit(train_frames)
        fit_s = time.time() - t0
        probs = {f.id: model.predict_proba(f.image) for f in test_frames}
    else:
        from dropseg.train import TrainConfig, predict_proba, train
        cfg = TrainConfig(arch=method, iters=iters, batch=4, crop=192, threads=8, seed=seed,
                          lr=1e-3 if method == "plain" else 3e-4)
        model, _ = train(train_frames, cfg, log=lambda *a: None)
        fit_s = time.time() - t0
        probs = {f.id: predict_proba(model, f.image) for f in test_frames}
    return probs, fit_s


def run_experiment(method, name, folds, frames_by_id, iters):
    out = OUT / method
    (out / "pred").mkdir(parents=True, exist_ok=True)
    for k, (train_ids, test_ids) in enumerate(folds):
        path = out / ("%s_fold%d.json" % (name, k))
        if path.is_file():
            continue
        train_frames = [frames_by_id[i] for i in train_ids]
        test_frames = [frames_by_id[i] for i in test_ids]
        probs, fit_s = fit_predict(method, train_frames, test_frames, iters, seed=k)
        per_frame = {}
        for f in test_frames:
            p = probs[f.id]
            cv2.imwrite(str(out / "pred" / ("%s__%s.png" % (name, f.id))), np.round(p * 255).astype(np.uint8))
            per_frame[f.id] = frame_scores(p >= 0.5, f.mask)
        path.write_text(json.dumps({"method": method, "experiment": name, "fold": k,
                                    "train": train_ids, "test": test_ids, "fit_s": round(fit_s, 1),
                                    "frames": per_frame}, indent=1))
        print("%-9s %-12s fold %d  dice %s  (%.0fs)" % (method, name, k, ", ".join(
            "%.3f" % v["dice"] for v in per_frame.values()), fit_s), flush=True)


def summarise(method, name, frames_by_id):
    files = sorted((OUT / method).glob("%s_fold*.json" % name))
    if not files:
        return None
    pairs, per = [], {}
    for f in files:
        d = json.loads(f.read_text())
        for fid, s in d["frames"].items():
            p = cv2.imread(str(OUT / method / "pred" / ("%s__%s.png" % (name, fid))), 0) >= 128
            pairs.append((p, frames_by_id[fid].mask))
            per[fid] = s
    dice = [s["dice"] for s in per.values()]
    return {"method": method, "experiment": name, "frames": len(per),
            "pooled": pooled_scores(pairs),
            "dice_mean": float(np.nanmean(dice)), "dice_std": float(np.nanstd(dice)),
            "per_frame": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["forest", "plain", "resnet18"])
    ap.add_argument("--iters", type=int, default=800)
    args = ap.parse_args()
    frames = load_frames()
    by_id = {f.id: f for f in frames}
    A = [f for f in frames if f.session == "A"]
    R = sorted((f for f in frames if f.session == "R"), key=lambda f: f.id)
    experiments = {
        "session_A": list(time_block_folds(A)),
        "A_to_R": [([f.id for f in A], [f.id for f in R])],
        "A_plus_R": [([f.id for f in A] + [f.id for f in R[1::2]], [f.id for f in R[0::2]]),
                     ([f.id for f in A] + [f.id for f in R[0::2]], [f.id for f in R[1::2]])],
    }
    for method in args.methods:
        for name, folds in experiments.items():
            run_experiment(method, name, folds, by_id, args.iters)
    summary = []
    for method in args.methods:
        for name in experiments:
            s = summarise(method, name, by_id)
            if s:
                summary.append(s)
                print("%-9s %-10s dice %.3f +- %.3f  pooled dice %.3f iou %.3f" % (
                    method, name, s["dice_mean"], s["dice_std"], s["pooled"]["dice"], s["pooled"]["iou"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
