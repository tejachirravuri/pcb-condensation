#!/usr/bin/env python3
"""Train the final U-Net on every labelled frame: models/unet_final.pt.

    python scripts/train_final.py [--arch resnet18] [--iters 1500]

If results/optuna.json exists, its best settings are used for the
architecture it found (learning rate, loss weight, crop size and so on);
otherwise the cross-validation settings.
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

from dropseg.data import load_frames  # noqa: E402
from dropseg.train import TrainConfig, save, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None, help="default: best architecture of the search, else resnet18")
    ap.add_argument("--iters", type=int, default=1500)
    args = ap.parse_args()
    params = {"arch": "resnet18", "lr": 3e-4}
    tuned = ROOT / "results" / "optuna.json"
    if tuned.is_file():
        params.update(json.loads(tuned.read_text())["best"])
    if args.arch and args.arch != params["arch"]:
        params = {"arch": args.arch, "lr": 1e-3 if args.arch == "plain" else 3e-4}
    cfg = TrainConfig(iters=args.iters, batch=4, threads=8, **params)
    frames = load_frames()
    print("training %s on %d frames: %s" % (cfg.arch, len(frames), params))
    t0 = time.time()
    model, hist = train(frames, cfg, log_every=250)
    path = ROOT / "models" / "unet_final.pt"
    save(model, cfg, path)
    (ROOT / "models" / "unet_final.json").write_text(json.dumps(
        {"config": cfg.__dict__, "frames": [f.id for f in frames], "history": hist,
         "train_minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
