#!/usr/bin/env python3
"""Collects the labelled frames and recordings into data/.

Only frames that were labelled by hand are used for training and testing.
The 500 "augmented" images of the 2022 work are not used: they were made
from 10 labelled frames and their red and blue channels are swapped.

    python scripts/build_dataset.py --source <project folder> --layout <layout.json>

The layout file says where in the project folder each part lies (see
scripts/source_layout.example.json): the Apeer images and masks, the
Labelme folders, the Apeer polygons of a65, the reference measurement and
the folder with the recordings a to j.

Result:

    data/labelled/images/<id>.png      original frame
    data/labelled/masks/<id>.png       0 = dry, 1 = water
    data/labelled/instances/<id>.json  droplet polygons (where available)
    data/labelled/index.csv            id, session, minute, size, label source
    data/recordings/<a..j>/<n>.jpg     unlabelled frames, one every 30 s
    data/raw_labels/...                the label files as exported
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

FRAME_INTERVAL_MIN = 0.5      # frames were extracted every 30 s

LAYOUT_KEYS = ("apeer", "labelme", "apeer_polygons", "reference", "recordings")

SIZE_CLASS = {"droplet_mini": "mini", "droplet_small": "small", "droplet_medium": "medium",
              "Water_small": "small", "Water_big": "medium", "droplet": "unknown"}


def frame_number(name):
    return int(re.findall(r"(\d+)", Path(name).stem)[0])


def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def polygons_from_labelme(path):
    data = load_json(path)
    out = []
    for s in data["shapes"]:
        if s.get("shape_type", "polygon") != "polygon" or len(s["points"]) < 3:
            continue
        out.append({"size_class": SIZE_CLASS.get(s["label"], "unknown"),
                    "points": [[round(x, 1), round(y, 1)] for x, y in s["points"]]})
    return out


def polygons_from_apeer(path):
    rows = load_json(path)
    header = rows[0]
    cls, geom = header.index("Class"), header.index("Geometry")
    out = []
    for r in rows[1:]:
        if r[cls] != "droplet" or len(r[geom]) < 3:
            continue
        out.append({"size_class": "unknown", "points": r[geom]})
    return out


def rasterise(polygons, shape):
    mask = np.zeros(shape[:2], np.uint8)
    for p in polygons:
        cv2.fillPoly(mask, [np.round(np.array(p["points"])).astype(np.int32)], 1)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--layout", type=Path, required=True, help="JSON with the sub-folders, see the example")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = ap.parse_args()
    src, out = args.source, args.out
    layout = json.loads(args.layout.read_text())
    missing = [k for k in LAYOUT_KEYS if k not in layout]
    if missing:
        raise SystemExit("layout file lacks %s" % ", ".join(missing))
    apeer_dir, labelme_dirs, apeer_poly_file = layout["apeer"], layout["labelme"], layout["apeer_polygons"]
    reference_dir, recordings_dir = layout["reference"], layout["recordings"]
    lab = out / "labelled"
    for sub in ("images", "masks", "instances"):
        (lab / sub).mkdir(parents=True, exist_ok=True)
    rows = []

    def add(fid, image, mask, session, minute, source, polygons=None):
        cv2.imwrite(str(lab / "images" / ("%s.png" % fid)), image)
        cv2.imwrite(str(lab / "masks" / ("%s.png" % fid)), mask.astype(np.uint8))
        if polygons is not None:
            (lab / "instances" / ("%s.json" % fid)).write_text(json.dumps(
                {"image": "%s.png" % fid, "droplets": polygons}))
        rows.append({"id": fid, "session": session, "minute": minute,
                     "width": image.shape[1], "height": image.shape[0],
                     "water_fraction": round(float(mask.mean()), 4), "label_source": source,
                     "instances": len(polygons) if polygons is not None else ""})

    # session A, recording a: droplet polygons (Labelme) for the early frames
    seen = set()
    for folder in labelme_dirs:
        for js in sorted((src / folder).glob("a*.json")):
            n = frame_number(js.name)
            if n in seen:
                continue
            data = load_json(js)
            if isinstance(data, list):
                continue
            seen.add(n)
            image = cv2.imread(str(src / recordings_dir / "a" / ("a%d.jpg" % n)))
            polys = polygons_from_labelme(js)
            add("A_a%03d" % n, image, rasterise(polys, image.shape), "A",
                (n - 1) * FRAME_INTERVAL_MIN, "labelme polygons", polys)

    # session A: water masks (Apeer brush tool), polygons for a65 as well
    apeer_poly = polygons_from_apeer(src / apeer_poly_file)
    for img in sorted((src / apeer_dir / "images").glob("a*.jpg"), key=lambda p: frame_number(p.name)):
        n = frame_number(img.name)
        if n in seen:
            continue
        image = cv2.imread(str(img))
        raw = cv2.imread(str(src / apeer_dir / "masks" / img.name), 0)
        mask = (raw < 128).astype(np.uint8)          # Apeer export: black = water
        add("A_a%03d" % n, image, mask, "A", (n - 1) * FRAME_INTERVAL_MIN, "apeer mask",
            apeer_poly if n == 65 else None)

    # session R: reference measurement Feb 2022, other camera setup, 512 px crops
    for img in sorted((src / reference_dir / "images").glob("*.jpg"), key=lambda p: frame_number(p.name)):
        n = frame_number(img.name)
        image = cv2.imread(str(img))
        raw = cv2.imread(str(src / reference_dir / "masks" / img.name), 0)
        mask = (raw >= 128).astype(np.uint8)         # ImageJ export: white = water
        add("R_a%03d" % n, image, mask, "R", None, "apeer mask (reference measurement)")

    with (lab / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("labelled frames:", len(rows))

    # unlabelled recordings
    rec_out = out / "recordings"
    total = 0
    for rec in sorted(p for p in (src / recordings_dir).iterdir() if p.is_dir() and len(p.name) == 1):
        dst = rec_out / rec.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in rec.glob("*.jpg"):
            target = dst / ("%s%03d.jpg" % (rec.name, frame_number(f.name)))
            if not target.exists():
                shutil.copyfile(f, target)
            total += 1
    print("recording frames:", total)

    # raw label files as exported, for reference
    raw = out / "raw_labels"
    for sub, pattern in [("labelme", (src / labelme_dirs[0]).glob("*.json")),
                         ("labelme", (src / labelme_dirs[1]).glob("a*.json")),
                         ("apeer_masks", (src / apeer_dir / "masks").glob("*.jpg")),
                         ("reference_masks", (src / reference_dir / "masks").glob("*.jpg"))]:
        (raw / sub).mkdir(parents=True, exist_ok=True)
        for f in pattern:
            shutil.copyfile(f, raw / sub / f.name)


if __name__ == "__main__":
    main()
