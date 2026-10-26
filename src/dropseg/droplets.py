"""Single droplets from a water mask, and their statistics.

Touching droplets merge into one region in a semantic mask. They are split
with a watershed on the distance transform: every local maximum of the
distance to the dry area that is far enough from the next one becomes the
centre of a droplet.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
from scipy import ndimage as ndi


def split_droplets(water: np.ndarray, min_area_px: int = 4, min_peak_distance: int = 3,
                   max_area_px: Optional[int] = None) -> np.ndarray:
    """Label image, one id per droplet. Regions larger than ``max_area_px``
    (water films) are kept whole."""
    water = water.astype(np.uint8)
    dist = cv2.distanceTransform(water, cv2.DIST_L2, 5)
    # peaks: maxima of the smoothed distance that rise above their surroundings
    smooth = cv2.GaussianBlur(dist, (0, 0), 1.0)
    size = 2 * min_peak_distance + 1
    peaks = (smooth == ndi.maximum_filter(smooth, size=size)) & (smooth > 1.0)
    markers, n = ndi.label(peaks)
    # every separate water region needs at least one seed
    comp, n_comp = ndi.label(water)
    if n_comp:
        seeded = np.zeros(n_comp + 1, bool)
        seeded[np.unique(comp[markers > 0])] = True
        for c in np.nonzero(~seeded[1:])[0] + 1:
            y, x = ndi.maximum_position(dist, comp, c)
            n += 1
            markers[y, x] = n
    # flood the inverted distance from the seeds (image foresting transform);
    # dry pixels are the highest ground, so floods do not cross them
    relief = np.round(1000 * (float(smooth.max()) - smooth)).clip(0, 65000).astype(np.uint16)
    relief[water == 0] = 65535
    labels = ndi.watershed_ift(relief, markers.astype(np.int32))
    labels[water == 0] = 0
    # merge back regions that are part of a film
    if max_area_px is not None:
        comp, _ = ndi.label(water)
        sizes = np.bincount(comp.ravel())
        big = np.isin(comp, np.nonzero(sizes > max_area_px)[0][1:])
        if big.any():
            labels[big] = comp[big] + labels.max()
    # drop specks
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    for i in ids[counts < min_area_px]:
        labels[labels == i] = 0
    return labels


def droplet_table(labels: np.ndarray, px_to_um: Optional[float] = None, scale_to_full: float = 1.0):
    """Area, equivalent diameter, centroid and nearest neighbour distance per droplet.

    ``scale_to_full`` converts from the processing scale back to pixels of the
    original frame (2.0 for frames processed at half size)."""
    ids = np.unique(labels[labels > 0])
    if not len(ids):
        return []
    rows = []
    cents = ndi.center_of_mass(labels > 0, labels, ids)
    areas = ndi.sum(labels > 0, labels, ids)
    for i, a, c in zip(ids, areas, cents):
        a_full = float(a) * scale_to_full ** 2
        d_full = 2 * np.sqrt(a_full / np.pi)
        row = {"id": int(i), "area_px": a_full, "diameter_px": d_full,
               "y": float(c[0]) * scale_to_full, "x": float(c[1]) * scale_to_full}
        if px_to_um:
            row["diameter_um"] = d_full * px_to_um
        rows.append(row)
    if len(rows) > 1:
        from scipy.spatial import cKDTree
        pts = np.array([[r["x"], r["y"]] for r in rows])
        d, _ = cKDTree(pts).query(pts, k=2)
        for r, nn in zip(rows, d[:, 1]):
            r["nearest_px"] = float(nn)
    return rows


def summary(rows, roi_area_px: float) -> Dict[str, float]:
    if not rows:
        return {"count": 0, "density_per_mpx": 0.0, "mean_diameter_px": float("nan"),
                "median_diameter_px": float("nan"), "p10_diameter_px": float("nan"),
                "p90_diameter_px": float("nan"),
                "max_diameter_px": float("nan"), "mean_nearest_px": float("nan")}
    d = np.array([r["diameter_px"] for r in rows])
    nn = np.array([r.get("nearest_px", np.nan) for r in rows])
    return {"count": len(rows), "density_per_mpx": len(rows) / (roi_area_px / 1e6),
            "mean_diameter_px": float(d.mean()), "median_diameter_px": float(np.median(d)),
            "p10_diameter_px": float(np.percentile(d, 10)), "p90_diameter_px": float(np.percentile(d, 90)),
            "max_diameter_px": float(d.max()),
            "mean_nearest_px": float(np.nanmean(nn)) if np.isfinite(nn).any() else float("nan")}


def shape_measures(labels: np.ndarray):
    """Per droplet id: width (twice the largest distance to its edge) and
    elongation (ratio of the long to the short axis of the fitted ellipse)."""
    ids = np.unique(labels[labels > 0])
    if not len(ids):
        return ids, np.zeros(0), np.zeros(0)
    edge = (labels > 0) & (cv2.erode(labels.astype(np.float32), np.ones((3, 3), np.uint8)) == labels)
    dist = cv2.distanceTransform(edge.astype(np.uint8), cv2.DIST_L2, 3) + 1
    width = 2 * np.asarray(ndi.maximum(dist, labels, ids))
    yy, xx = np.indices(labels.shape)
    one = labels > 0
    n = np.asarray(ndi.sum(one, labels, ids))
    mx, my = (np.asarray(ndi.sum(v, labels, ids)) / n for v in (xx, yy))
    sxx = np.asarray(ndi.sum(xx * xx, labels, ids)) / n - mx ** 2
    syy = np.asarray(ndi.sum(yy * yy, labels, ids)) / n - my ** 2
    sxy = np.asarray(ndi.sum(xx * yy, labels, ids)) / n - mx * my
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    root = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + root, np.maximum(tr / 2 - root, 1e-3)
    return ids, width, np.sqrt(l1 / l2)


def remove_slivers(labels: np.ndarray, max_elongation: float = 3.0, min_width_px: float = 8.0,
                   max_any: float = 10.0) -> np.ndarray:
    """Drop long, thin regions. Droplets are round; the thin bright or dark
    lines along the edges of the comb fingers are not, but the network
    sometimes takes them for water. A region is dropped if it is more than
    ``max_elongation`` times longer than wide and narrower than
    ``min_width_px`` (pixels of the label image), or more than ``max_any``
    times longer than wide whatever its width."""
    ids, width, elong = shape_measures(labels)
    bad = ids[((elong > max_elongation) & (width < min_width_px)) | (elong > max_any)]
    if not len(bad):
        return labels
    out = labels.copy()
    out[np.isin(labels, bad)] = 0
    return out
