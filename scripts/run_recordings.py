#!/usr/bin/env python3
"""Droplet formation and growth in the ten climate chamber recordings.

    python scripts/run_recordings.py [--model models/unet_final.pt] [--no-video]
    python scripts/run_recordings.py --maskrcnn models/maskrcnn_final.pt --recordings a

One frame every 30 s. For each recording the droplets on the comb structure
are segmented, measured and tracked from frame to frame (see
dropseg/monitor.py and dropseg/tracking.py).

With --maskrcnn the droplets come straight from Mask R-CNN on the full
size frames instead of U-Net + watershed on half size frames (about 10 s
per frame on a CPU instead of 0.7 s), and the output goes to
results/recordings_maskrcnn/.

Writes to results/recordings/:
    <rec>.csv               one row per frame
    <rec>.png               count, size, merges and size distribution over time
    <rec>.mp4               frames with droplet outlines coloured by size
    <rec>_growth.jpg        the droplet that grew most, at six moments
    summary.json            events and growth numbers per recording
"""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from dropseg.data import recording_frames, recordings  # noqa: E402
from dropseg.droplets import shape_measures  # noqa: E402
from dropseg.monitor import SCALE, analyse, growth_strip  # noqa: E402
from dropseg.tracking import strongest_growth  # noqa: E402
from dropseg.train import load, predict_proba  # noqa: E402

OUT = ROOT / "results" / "recordings"
BINS = np.geomspace(10, 640, 13)        # diameter bins, full frame pixels


def size_histograms(labels, scale):
    out = []
    for lab in labels:
        a = np.bincount(lab.ravel())[1:]
        d = 2 * np.sqrt(a[a > 0] / np.pi) / scale
        out.append(np.histogram(d, BINS)[0])
    return np.array(out).T


def round_droplet(labels, tr, max_elongation=1.6):
    """Example droplets should look like droplets at the end of their track."""
    lab = (labels[tr.end] == tr.label_ids[-1]).astype(np.int32)
    _, _, elong = shape_measures(lab)
    return bool(len(elong)) and elong[0] <= max_elongation and tr.diameters()[0] >= 12


def plot(rec, rows, ev, hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["minute"] for r in rows])
    fig, ax = plt.subplots(4, 1, figsize=(7.5, 9.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 0.8, 1.3]})
    ax[0].plot(t, [r["count"] for r in rows], color="#333333")
    ax[0].set_ylabel("droplets")
    a2 = ax[0].twinx()
    a2.plot(t, [100 * r["coverage"] for r in rows], color="#1f77b4", alpha=0.7)
    a2.set_ylabel("water on combs (%)", color="#1f77b4")
    ax[1].plot(t, [r["median_diameter_px"] for r in rows], color="#333333", label="median")
    ax[1].plot(t, [r["p90_diameter_px"] for r in rows], color="#9467bd", label="90th percentile")
    ax[1].set_ylabel("droplet\ndiameter (px)")
    ax[1].legend(loc="upper left", fontsize=8)
    w = 0.4 * (t[1] - t[0]) if len(t) > 1 else 0.2
    ax[2].bar(t - w / 2, [r["new"] for r in rows], width=w, color="#2ca02c", label="new")
    ax[2].bar(t + w / 2, [r["merged"] for r in rows], width=w, color="#d62728", label="merges")
    ax[2].set_ylabel("per frame")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[3].imshow(hist, aspect="auto", origin="lower", cmap="Greys",
                 extent=[t[0], t[-1], 0, len(BINS) - 1])
    ticks = np.arange(0, len(BINS), 3)
    ax[3].set_yticks(ticks)
    ax[3].set_yticklabels(["%.0f" % BINS[i] for i in ticks])
    ax[3].set_ylabel("diameter (px)\nnumber of droplets")
    ax[3].set_xlabel("minutes from start of recording")
    for a in ax:
        for key, c in (("peak_count_min", "#2ca02c"), ("peak_coverage_min", "#1f77b4")):
            if ev[key] is not None:
                a.axvline(ev[key], color=c, ls="--", lw=1)
    for a in ax[:3]:
        a.grid(alpha=0.3)
    ax[0].set_title("recording %s: droplet formation and growth" % rec)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "unet_final.pt"))
    ap.add_argument("--recordings", nargs="*")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--maskrcnn", help="Mask R-CNN checkpoint; droplets from it instead of the U-Net")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    if args.maskrcnn:
        from dropseg.models import maskrcnn
        model, _ = maskrcnn.load(args.maskrcnn)
        out_dir, scale, name = ROOT / "results" / "recordings_maskrcnn", 1.0, Path(args.maskrcnn).name
        kw = dict(scale=1.0, instances=True)
        predict = lambda im: maskrcnn.predict(model, im)[0]  # noqa: E731
    else:
        model, _ = load(args.model)
        out_dir, scale, name = OUT, SCALE, Path(args.model).name
        kw = {}
        predict = lambda im: predict_proba(model, im)  # noqa: E731
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for rec in args.recordings or recordings():
        res = analyse(recording_frames(rec), predict, render=not args.no_video, **kw)
        rows, ev = res["rows"], {"recording": rec, **res["events"]}
        out.append(ev)
        with (out_dir / ("%s.csv" % rec)).open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        plot(rec, rows, ev, size_histograms(res["labels"], scale), out_dir / ("%s.png" % rec))
        tr = strongest_growth(res["tracks"], accept=lambda t: round_droplet(res["labels"], t))
        if tr is not None:
            d = tr.diameters()
            ev["example_track"] = {"start_min": rows[tr.start]["minute"], "frames": len(d),
                                   "diameter_start_px": round(float(d[0]), 1),
                                   "diameter_end_px": round(float(d[-1]), 1), "absorbed": tr.absorbed}
            cv2.imwrite(str(out_dir / ("%s_growth.jpg" % rec)),
                        growth_strip(res["images"], res["labels"], tr, pad=int(24 * scale)),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        if res["rendered"]:
            vw = cv2.VideoWriter(str(out_dir / ("%s.mp4" % rec)), cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 480))
            for fr in res["rendered"]:
                vw.write(cv2.resize(fr, (640, 480), interpolation=cv2.INTER_AREA))
            vw.release()
        phase = ("count %d -> %d, p90 %.0f -> %.0f px, %d merges in %.1f min" % (
            ev["peak_count"], ev["count_at_peak_coverage"], ev["p90_at_peak_count"], ev["p90_at_peak_coverage"],
            ev["merges_in_phase"], ev["merge_phase_min"]) if "count_ratio" in ev else "no growth phase (drying only)")
        print("%s  %3d frames  most droplets %3d at %5.1f min  wettest %3.0f%% at %5.1f min  %s  "
              "(%.2f s/frame)" % (rec, ev["frames"], ev["peak_count"], ev["peak_count_min"],
                                  100 * ev["peak_coverage"], ev["peak_coverage_min"], phase,
                                  ev["seconds_per_frame"]), flush=True)
    (out_dir / "summary.json").write_text(json.dumps({"model": name, "recordings": out}, indent=1))


if __name__ == "__main__":
    main()
