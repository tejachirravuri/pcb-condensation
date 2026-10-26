#!/usr/bin/env python3
"""Droplet growth as seen in the hand labels alone (no model involved).

    python scripts/plot_label_growth.py

Recording a has every droplet outlined at 9.5, 12, 14.5, 17 and 32 min.
Writes results/label_growth.json, label_growth.png (count and size over
time, size distributions) and label_growth_crops.jpg (one part of the comb
structure at each of these moments, droplet outlines in magenta).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dropseg.data import load_frame, read_index  # noqa: E402
from dropseg.metrics import rasterise  # noqa: E402

FRAMES = ["A_a020", "A_a025", "A_a030", "A_a035", "A_a065"]
CROP = (300, 300, 380, 380)          # y, x, h, w in the full frame


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = {r["id"]: r for r in read_index()}
    out, diams, tiles = [], [], []
    for fid in FRAMES:
        f = load_frame(rows[fid], scale=1.0)
        lab = rasterise(f.instances, f.mask.shape)
        a = np.bincount(lab.ravel())[1:]
        d = 2 * np.sqrt(a[a > 0] / np.pi)
        diams.append(d)
        out.append({"frame": fid, "minute": f.minute, "droplets": int(len(d)),
                    "coverage": float((lab > 0).mean()),
                    "median_diameter_px": float(np.median(d)), "p10_diameter_px": float(np.percentile(d, 10)),
                    "p90_diameter_px": float(np.percentile(d, 90)), "max_diameter_px": float(d.max()),
                    "label_source": f.label_source})
        y, x, h, w = CROP
        crop = cv2.cvtColor(f.image[y:y + h, x:x + w], cv2.COLOR_RGB2BGR).copy()
        sub = lab[y:y + h, x:x + w]
        k = np.ones((3, 3), np.uint8)
        s32 = sub.astype(np.float32)
        edge = ((cv2.dilate(s32, k) != s32) | (cv2.erode(s32, k) != s32)) & (sub > 0)
        crop[edge] = (255, 0, 255)
        cv2.putText(crop, "%.1f min" % f.minute, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(crop)
    cv2.imwrite(str(ROOT / "results" / "label_growth_crops.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])

    t = [r["minute"] for r in out]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(t, [r["droplets"] for r in out], "o-", color="#333333")
    ax[0].set_ylabel("droplets in the frame")
    ax[0].set_xlabel("minutes")
    ax[1].plot(t, [r["median_diameter_px"] for r in out], "o-", color="#333333", label="median")
    ax[1].plot(t, [r["p90_diameter_px"] for r in out], "o-", color="#9467bd", label="90th percentile")
    ax[1].plot(t, [r["max_diameter_px"] for r in out], "o--", color="#9467bd", alpha=0.5, label="largest")
    ax[1].set_ylabel("diameter (px)")
    ax[1].set_xlabel("minutes")
    ax[1].legend(fontsize=8)
    bins = np.geomspace(8, 512, 19)
    for r, d, c in zip(out, diams, plt.cm.viridis(np.linspace(0, 0.9, len(out)))):
        h, _ = np.histogram(d, bins)
        ax[2].step(bins[:-1], h, where="post", color=c, label="%.1f min" % r["minute"])
    ax[2].set_xscale("log")
    ax[2].set_xlabel("diameter (px)")
    ax[2].set_ylabel("droplets")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Recording a, hand labels: fewer, larger droplets over time")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "label_growth.png", dpi=110)
    (ROOT / "results" / "label_growth.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print("%5.1f min  %3d droplets  median %3.0f px  p90 %3.0f px  max %3.0f px  coverage %.2f" % (
            r["minute"], r["droplets"], r["median_diameter_px"], r["p90_diameter_px"],
            r["max_diameter_px"], r["coverage"]))


if __name__ == "__main__":
    main()
