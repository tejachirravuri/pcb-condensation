"""Paired image/mask augmentation, applied on the fly to training crops only.

The 2022 dataset was made by augmenting 10 frames into 500 files before the
train/validation split, so augmented copies of the same frame ended up on
both sides. Here augmentation only ever happens inside the training loop.
"""
from __future__ import annotations

import cv2
import numpy as np


def random_crop(image, mask, size, rng, water_bias=0.5):
    """Crop of size x size. With probability ``water_bias`` the crop is
    centred near a water pixel, so sparse droplets are seen often enough."""
    H, W = mask.shape
    if H < size or W < size:
        pad_h, pad_w = max(0, size - H), max(0, size - W)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
        H, W = mask.shape
    if rng.random() < water_bias and mask.any():
        ys, xs = np.nonzero(mask)
        i = rng.integers(len(ys))
        y0 = int(np.clip(ys[i] - rng.integers(size), 0, H - size))
        x0 = int(np.clip(xs[i] - rng.integers(size), 0, W - size))
    else:
        y0, x0 = int(rng.integers(0, H - size + 1)), int(rng.integers(0, W - size + 1))
    return image[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size]


def augment(image, mask, rng):
    """Geometric changes for both, photometric changes for the image only."""
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[::-1], mask[::-1]
    k = int(rng.integers(4))
    image, mask = np.rot90(image, k), np.rot90(mask, k)
    image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)

    if rng.random() < 0.5:   # small rotation and zoom
        h, w = mask.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.85, 1.2))
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    img = image.astype(np.float32)
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)            # contrast, brightness
    if rng.random() < 0.5:                                                 # gamma
        img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.3:                                                 # colour cast
        img = img * rng.uniform(0.9, 1.1, size=3)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask
