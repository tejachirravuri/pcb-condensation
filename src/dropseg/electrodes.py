"""The comb structure: two interdigitated electrodes.

The test structure is a pair of comb electrodes whose fingers interlock.
Droplets are measured inside this structure only.

The camera does not move during a recording, so the electrodes are found
once on a dry frame: metal is bright and bluish (B > R) where the solder
mask is yellow, and the two combs are the two largest metal regions.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Electrodes:
    labels: np.ndarray        # 0 background, 1 and 2 the two combs
    roi: np.ndarray           # bool, area of the comb structure (convex hull of both)
    finger_width_px: float    # median width of the fingers, a ruler for the frame

    @property
    def shape(self):
        return self.labels.shape


def metal_mask(image_rgb: np.ndarray, min_blue_excess=15, min_value=110) -> np.ndarray:
    r = image_rgb[..., 0].astype(int)
    b = image_rgb[..., 2].astype(int)
    v = image_rgb.max(axis=2)
    metal = ((b - r) > min_blue_excess) & (v > min_value)
    metal = cv2.morphologyEx(metal.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(metal, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def find_electrodes(dry_image_rgb: np.ndarray) -> Electrodes:
    metal = metal_mask(dry_image_rgb)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(metal, 8)
    if n < 3:
        raise ValueError("could not find two electrodes")
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    labels = np.zeros(metal.shape, np.uint8)
    labels[lab == order[0]] = 1
    labels[lab == order[1]] = 2
    pts = cv2.findNonZero((labels > 0).astype(np.uint8))
    roi = np.zeros(metal.shape, np.uint8)
    cv2.fillConvexPoly(roi, cv2.convexHull(pts), 1)
    # finger width: twice the median distance to the background on the skeleton-ish ridge
    dist = cv2.distanceTransform((labels > 0).astype(np.uint8), cv2.DIST_L2, 5)
    ridge = dist[dist >= np.percentile(dist[dist > 0], 60)]
    return Electrodes(labels, roi.astype(bool), float(2 * np.median(ridge)))
