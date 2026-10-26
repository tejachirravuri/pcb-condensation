"""Following condensation droplets through a sequence of frames.

Every frame: the water mask from the network, split into droplets inside
the comb structure; count, size and spacing of the droplets. Over the
sequence the droplets are tracked (dropseg.tracking), which shows how
condensation develops: droplets appear, grow, and merge into fewer, larger
ones. Events of a sequence:

    onset          first frame with >= 1 % of the comb area wet, confirmed
                   by the next frame
    peak count     the frame with the most droplets before the wettest
                   frame: the end of nucleation
    peak coverage  the wettest frame
    merge phase    from peak count to peak coverage: the number of droplets
                   falls while the wet area still grows, so the droplets
                   grow by merging; measured by how much the count drops
                   and the 90th percentile diameter rises
    dry again      first frame after the peak with less than a quarter of
                   the peak coverage

Counts and coverage are smoothed with a running median over five frames
(2.5 min) before peaks are picked, so a single noisy frame cannot set an
event.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from .data import frame_minute
from .droplets import droplet_table, remove_slivers, split_droplets, summary
from .electrodes import Electrodes, find_electrodes
from .tracking import track, track_summary

SCALE = 0.5          # frames are processed at half size, like the training frames
WET = 0.01           # fraction of the comb area


def read_frame(path, scale=SCALE) -> np.ndarray:
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def first_confirmed(flags: Sequence[bool]) -> Optional[int]:
    for i in range(len(flags) - 1):
        if flags[i] and flags[i + 1]:
            return i
    return None


def size_colour(d_px):
    """BGR colour from small (cyan) to large (magenta) droplets, by diameter."""
    f = float(np.clip(np.log(max(d_px, 1) / 10) / np.log(30), 0, 1))     # 10 px .. 300 px
    return (255, int(255 * (1 - f)), int(255 * f))


def overlay(rgb, water, droplets=None, electrodes: Optional[Electrodes] = None, scale=SCALE) -> np.ndarray:
    """BGR image: water tinted blue, droplet outlines coloured by size."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    tint = out.copy()
    tint[water > 0] = (255, 120, 0)
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
    if electrodes is not None:
        roi = cv2.morphologyEx(electrodes.roi.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        out[roi > 0] = (0, 200, 0)
    if droplets is not None and droplets.max():
        edge = (cv2.dilate(droplets.astype(np.float32), np.ones((3, 3), np.uint8)) !=
                cv2.erode(droplets.astype(np.float32), np.ones((3, 3), np.uint8))) & (droplets > 0)
        ids = np.unique(droplets[droplets > 0])
        areas = np.bincount(droplets.ravel())
        lut = np.zeros((int(ids.max()) + 1, 3), np.uint8)
        for i in ids:
            lut[i] = size_colour(2 * np.sqrt(areas[i] / np.pi) / scale)
        out[edge] = lut[droplets[edge]]
    return out


def electrodes_ok(el: Electrodes) -> bool:
    """Two combs of similar size, each a fair part of the frame."""
    areas = np.bincount(el.labels.ravel(), minlength=3)[1:]
    return bool(areas.min() > 0.03 * el.labels.size and areas.min() / areas.max() > 0.6)


def frame_stats(water, el: Electrodes, scale=SCALE, droplets=None):
    """Row of numbers for one frame, and its droplet label image. Without
    ``droplets`` (a label image from an instance model) the water mask is
    split into droplets with a watershed. Thin lines along the finger edges
    are removed (droplets.remove_slivers); coverage is the area of the
    remaining droplets."""
    roi_px = el.roi.sum() / scale ** 2
    if droplets is None:
        lab = split_droplets(water * el.roi, min_area_px=max(1, round(12 * scale ** 2)),
                             min_peak_distance=max(1, round(10 * scale)))
    else:
        lab = droplets * el.roi
    lab = remove_slivers(lab, min_width_px=16 * scale)
    s = summary(droplet_table(lab, scale_to_full=1 / scale), roi_px)
    row = {"coverage": float((lab > 0)[el.roi].mean())}
    row.update({k: s[k] for k in ("count", "density_per_mpx", "median_diameter_px", "mean_diameter_px",
                                  "p10_diameter_px", "p90_diameter_px", "max_diameter_px",
                                  "mean_nearest_px")})
    return row, lab


def smooth(x, k=5):
    """Running median over ``k`` frames, so single noisy frames do not pick a peak."""
    x = np.asarray(x, float)
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.array([np.nanmedian(pad[i:i + k]) for i in range(len(x))])


def events(rows: List[dict], wet=WET) -> dict:
    m = lambda i: None if i is None else rows[i]["minute"]  # noqa: E731
    cov = smooth([r["coverage"] for r in rows])
    cnt = smooth([r["count"] for r in rows])
    p90 = smooth([r["p90_diameter_px"] for r in rows])
    i_on = first_confirmed([r["coverage"] >= wet for r in rows])
    i_cov = int(np.argmax(cov))
    i_cnt = int(np.argmax(cnt[:i_cov + 1]))           # most droplets before the wettest frame
    i_dry = next((i for i in range(i_cov, len(rows)) if cov[i] < 0.25 * cov[i_cov]), None)
    ev = {"frames": len(rows), "minutes": rows[-1]["minute"],
          "starts_wet": bool(len(rows) > 1 and rows[0]["coverage"] >= wet and rows[1]["coverage"] >= wet),
          "onset_min": m(i_on),
          "peak_count": int(round(cnt[i_cnt])), "peak_count_min": m(i_cnt),
          "peak_coverage": float(cov[i_cov]), "peak_coverage_min": m(i_cov),
          "dry_again_min": m(i_dry),
          "total_merges": int(sum(r.get("merged", 0) for r in rows)),
          "total_new": int(sum(r.get("new", 0) for r in rows[1:]))}
    # growth by merging: from the most droplets to the wettest frame
    phase = i_cov - i_cnt
    ev["merge_phase_min"] = phase * (rows[1]["minute"] - rows[0]["minute"]) if len(rows) > 1 else 0.0
    if phase >= 4:
        ev.update({"count_at_peak_coverage": int(round(cnt[i_cov])),
                   "count_ratio": float(cnt[i_cov] / max(cnt[i_cnt], 1)),
                   "p90_at_peak_count": float(p90[i_cnt]), "p90_at_peak_coverage": float(p90[i_cov]),
                   "p90_ratio": float(p90[i_cov] / max(p90[i_cnt], 1)),
                   "merges_in_phase": int(sum(r.get("merged", 0) for r in rows[i_cnt + 1:i_cov + 1])),
                   "new_in_phase": int(sum(r.get("new", 0) for r in rows[i_cnt + 1:i_cov + 1]))})
    return ev


def analyse(paths: Sequence[Path], predict: Callable[[np.ndarray], np.ndarray],
            interval_min: float = 0.5, render: bool = False, scale=SCALE, instances=False):
    """Run ``predict`` (RGB frame -> water probability) over a sequence.
    With ``instances``, ``predict`` returns a droplet label image instead
    (Mask R-CNN), and the watershed is not needed.

    The comb structure is located on the driest frame of the sequence; the
    camera has to stay in place for the whole sequence. Returns a dict with
    rows (one per frame), events, tracks, electrodes, the droplet label
    images, and the rendered frames if ``render``."""
    t0 = time.time()
    images, water, given = [], [], []
    for p in paths:
        im = read_frame(p, scale)
        images.append(im)
        out = predict(im)
        if instances:
            given.append(out.astype(np.uint16))
            water.append((out > 0).astype(np.uint8))
        else:
            water.append((out >= 0.5).astype(np.uint8))
    sec = (time.time() - t0) / max(len(paths), 1)
    dry = int(np.argmin([w.mean() for w in water]))
    el = find_electrodes(images[dry])
    ok = electrodes_ok(el)
    rows, labels = [], []
    for k, (p, w) in enumerate(zip(paths, water)):
        row, lab = frame_stats(w, el, scale, given[k] if instances else None)
        rows.append({"frame": Path(p).stem, "minute": frame_minute(p, interval_min), **row})
        labels.append(lab)
    tracks, per_frame = track(labels, scale_to_full=1 / scale, interval_min=interval_min)
    for row, f in zip(rows, per_frame):
        row.update(f)
    ev = events(rows)
    ev.update(track_summary(tracks, interval_min))
    ev.update({"electrodes_found": ok, "dry_reference": Path(paths[dry]).stem,
               "finger_width_px": round(el.finger_width_px / scale, 1), "seconds_per_frame": round(sec, 3)})
    rendered = []
    if render:
        for im, w, lab, row in zip(images, water, labels, rows):
            fr = overlay(im, w, lab, el, scale)
            cv2.putText(fr, "%s  %.1f min  droplets %d  median %.0f px  merges %d" % (
                row["frame"], row["minute"], row["count"], np.nan_to_num(row["median_diameter_px"]),
                row["merged"]), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            rendered.append(fr)
    return {"rows": rows, "events": ev, "tracks": tracks, "electrodes": el, "labels": labels,
            "images": images, "rendered": rendered}


def growth_strip(images, labels, tr, interval_min=0.5, panels=6, pad=12, size=160) -> np.ndarray:
    """Image strip of one tracked droplet at evenly spaced moments of its life."""
    # box around every place the droplet has been, since a merged droplet
    # carries on the track of the larger one that went into it
    ys, xs = [], []
    for k, i in enumerate(tr.label_ids):
        y, x = np.nonzero(labels[tr.start + k] == i)
        ys.append(y)
        xs.append(x)
    ys, xs = np.concatenate(ys), np.concatenate(xs)
    last = labels[tr.end]
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, last.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, last.shape[1])
    steps = np.unique(np.linspace(0, len(tr.areas) - 1, panels).round().astype(int))
    tiles = []
    for k in steps:
        t = tr.start + k
        crop = cv2.cvtColor(images[t][y0:y1, x0:x1], cv2.COLOR_RGB2BGR).copy()
        m = (labels[t][y0:y1, x0:x1] == tr.label_ids[k]).astype(np.uint8)
        h, w = crop.shape[:2]
        f = size / max(h, w)
        crop = cv2.resize(crop, (int(w * f), int(h * f)), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, cnts, -1, (255, 0, 255), 2)
        tile = np.full((size + 22, size, 3), 255, np.uint8)
        tile[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(tile, "+%.1f min" % (k * interval_min), (4, size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(tile)
    return np.hstack(tiles)
