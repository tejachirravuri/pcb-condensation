#!/usr/bin/env python3
"""Train Mask R-CNN on all five frames with outlined droplets.

    python scripts/train_maskrcnn_final.py [--iters 400]

Same settings as in scripts/run_instances.py, now on a20, a25, a30, a35
and a65 together. Saves models/maskrcnn_final.pt (half precision).
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from dropseg.data import load_frame, read_index  # noqa: E402
from dropseg.models import maskrcnn  # noqa: E402

FRAMES = ["A_a020", "A_a025", "A_a030", "A_a035", "A_a065"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()
    rows = {r["id"]: r for r in read_index()}
    frames = [load_frame(rows[i], scale=1.0) for i in FRAMES]
    cfg = maskrcnn.MRCNNConfig(iters=args.iters, threads=8)
    t0 = time.time()
    model, hist = maskrcnn.train(frames, cfg)
    path = ROOT / "models" / "maskrcnn_final.pt"
    maskrcnn.save(model, cfg, path)
    (ROOT / "models" / "maskrcnn_final.json").write_text(json.dumps(
        {"config": cfg.__dict__, "frames": FRAMES, "history": hist,
         "train_minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
