#!/usr/bin/env python3
"""Why the 2022 Dice of 0.90 was too optimistic.

    python scripts/run_leaky_protocol.py [--iters 800]

The 2022 dataset: 10 labelled frames (a45 to a90) augmented into 500 crops,
then split at random 80/20 into training and validation. Crops of the same
frame end up on both sides, so the validation score measures how well the
network remembers those 10 frames.

This script repeats that protocol with the plain U-Net and then scores the
same network on frames it has never seen (a20 to a35 of the same recording,
and the reference measurement R). The honest numbers to compare with are the
time-block cross-validation results of scripts/run_cv.py.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from dropseg.augment import augment, random_crop  # noqa: E402
from dropseg.data import Frame, load_frames  # noqa: E402
from dropseg.metrics import frame_scores, pooled_scores  # noqa: E402
from dropseg.train import TrainConfig, predict_proba, train  # noqa: E402

OUT = ROOT / "results" / "leaky_protocol.json"
SOURCE = ["A_a%03d" % m for m in range(45, 91, 5)]
UNSEEN = ["A_a020", "A_a025", "A_a030", "A_a035"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--copies", type=int, default=50)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    frames = {f.id: f for f in load_frames()}

    crops = []
    for fid in SOURCE:
        f = frames[fid]
        for k in range(args.copies):
            im, m = random_crop(f.image, f.mask, 192, rng, water_bias=0.0)
            im, m = augment(im, m, rng)
            crops.append(Frame("%s_%02d" % (fid, k), "A", None, im, m))
    order = rng.permutation(len(crops))
    n_val = len(crops) // 5
    val = [crops[i] for i in order[:n_val]]
    tr = [crops[i] for i in order[n_val:]]
    print("%d crops from %d frames: %d train, %d validation" % (len(crops), len(SOURCE), len(tr), len(val)))

    cfg = TrainConfig(arch="plain", lr=1e-3, iters=args.iters, batch=4, crop=192, threads=8)
    model, hist = train(tr, cfg, log=print, log_every=200)

    def score(fs):
        pairs = [(predict_proba(model, f.image) >= 0.5, f.mask) for f in fs]
        return {"pooled": pooled_scores(pairs),
                "dice_mean": float(np.mean([frame_scores(p, t)["dice"] for p, t in pairs]))}

    res = {"protocol": "10 frames -> %d augmented crops -> random 80/20 split" % len(crops),
           "config": cfg.__dict__, "history": hist,
           "random_split_validation": score(val),
           "unseen_frames_same_recording": score([frames[i] for i in UNSEEN]),
           "reference_measurement": score([f for f in frames.values() if f.session == "R"])}
    for k in ("random_split_validation", "unseen_frames_same_recording", "reference_measurement"):
        print("%-30s dice %.3f (pooled %.3f)" % (k, res[k]["dice_mean"], res[k]["pooled"]["dice"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
