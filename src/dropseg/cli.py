"""Command line.

    dropseg predict FRAME [FRAME ...] --model models/unet_final.pt --out out/
    dropseg watch FOLDER --model models/unet_final.pt --out out/ [--interval 0.5]

``predict`` writes the water mask and an overlay per frame and prints the
coverage and droplet numbers. ``watch`` treats a folder of frames as one
recording (sorted by the number in the file name) and writes the droplet
time series and the events of dropseg.monitor.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def _model(path, threads):
    import torch
    from .train import load, predict_proba
    if threads:
        torch.set_num_threads(threads)
    model, _ = load(path)
    return lambda im: predict_proba(model, im)


def cmd_predict(args):
    from .droplets import droplet_table, split_droplets, summary
    from .monitor import overlay, read_frame
    predict = _model(args.model, args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in args.frames:
        im = read_frame(p, args.scale)
        water = (predict(im) >= 0.5).astype(np.uint8)
        lab = split_droplets(water, min_area_px=3, min_peak_distance=5)
        s = summary(droplet_table(lab, scale_to_full=1 / args.scale), water.size / args.scale ** 2)
        stem = Path(p).stem
        cv2.imwrite(str(out / ("%s_mask.png" % stem)), water * 255)
        cv2.imwrite(str(out / ("%s_overlay.jpg" % stem)), overlay(im, water, lab, scale=args.scale))
        print("%s  water %.1f%%  droplets %d  median diameter %.1f px" % (
            stem, 100 * water.mean(), s["count"], s["median_diameter_px"]))


def cmd_watch(args):
    from .monitor import analyse
    predict = _model(args.model, args.threads)
    files = [p for p in Path(args.folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")]
    files.sort(key=lambda p: int((re.findall(r"(\d+)", p.stem) or ["0"])[-1]))
    if len(files) < 2:
        sys.exit("need at least two frames in %s" % args.folder)
    res = analyse(files, predict, interval_min=args.interval)
    rows, ev = res["rows"], res["events"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "timeseries.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "events.json").write_text(json.dumps(ev, indent=1))
    print("most droplets: %d at %s min, wettest: %.0f %% at %s min, %d merges in all" % (
        ev["peak_count"], ev["peak_count_min"], 100 * ev["peak_coverage"], ev["peak_coverage_min"],
        ev["total_merges"]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dropseg", description="Condensation on PCB comb structures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("predict", "watch"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="models/unet_final.pt")
        p.add_argument("--out", default="out")
        p.add_argument("--threads", type=int, default=0)
        if name == "predict":
            p.add_argument("frames", nargs="+")
            p.add_argument("--scale", type=float, default=0.5,
                           help="resize factor before the network; 0.5 for 1280x960 frames")
        else:
            p.add_argument("folder")
            p.add_argument("--interval", type=float, default=0.5, help="minutes between frames")
    args = ap.parse_args(argv)
    {"predict": cmd_predict, "watch": cmd_watch}[args.cmd](args)


if __name__ == "__main__":
    main()
