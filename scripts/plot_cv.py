#!/usr/bin/env python3
"""Figures for the cross-validation results.

    python scripts/plot_cv.py

results/cv_examples.jpg   held-out frames: image, hand label, forest,
                          plain U-Net, ResNet-18 U-Net (water in blue)
results/cv_per_frame.png  Dice per held-out frame of recording a
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dropseg.data import load_frames  # noqa: E402

CV = ROOT / "results" / "cv"
METHODS = [("forest", "pixel forest"), ("plain", "plain U-Net"), ("resnet18", "ResNet-18 U-Net")]
EXAMPLES = [("session_A", "A_a020"), ("session_A", "A_a050"), ("session_A", "A_a085"), ("A_plus_R", "R_a012")]


def tint(rgb, mask):
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    blue = out.copy()
    blue[mask > 0] = (255, 120, 0)
    return cv2.addWeighted(blue, 0.5, out, 0.5, 0)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main():
    frames = {f.id: f for f in load_frames()}
    rows = []
    for exp, fid in EXAMPLES:
        f = frames[fid]
        h, w = 300, int(300 * f.image.shape[1] / f.image.shape[0])
        tiles = [label(cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR), fid), label(tint(f.image, f.mask), "hand label")]
        for m, name in METHODS:
            p = cv2.imread(str(CV / m / "pred" / ("%s__%s.png" % (exp, fid))), 0)
            tiles.append(label(tint(f.image, p >= 128), name))
        rows.append(np.hstack([cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA) for t in tiles]))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(str(ROOT / "results" / "cv_examples.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = {(s["method"], s["experiment"]): s for s in json.loads((CV / "summary.json").read_text())}
    ids = sorted(summary[("resnet18", "session_A")]["per_frame"], key=lambda i: frames[i].minute)
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for k, ((m, name), c) in enumerate(zip(METHODS, ["#999999", "#6baed6", "#08519c"])):
        d = [summary[(m, "session_A")]["per_frame"][i]["dice"] for i in ids]
        ax.bar(x + (k - 1) * 0.27, d, width=0.27, color=c, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%.1f min" % (i[2:], frames[i].minute) for i in ids], fontsize=8)
    ax.set_ylabel("Dice (held out)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Recording a, time-block cross-validation")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "cv_per_frame.png", dpi=110)


if __name__ == "__main__":
    main()
