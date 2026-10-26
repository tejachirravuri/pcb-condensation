"""Classical baseline: random forest on hand made per-pixel features.

This is the kind of pixel classifier that interactive tools like ilastik
use. Features per pixel: colour (RGB and HSV), Gaussian blur, gradient
magnitude, Laplacian and local standard deviation at several scales. Water
changes the local texture (droplet rims, highlights), so the texture
features matter more than colour.
"""
from __future__ import annotations

import cv2
import numpy as np

SIGMAS = (1.0, 2.0, 4.0, 8.0)


def pixel_features(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    feats = [img[..., 0], img[..., 1], img[..., 2], hsv[..., 0], hsv[..., 1], hsv[..., 2]]
    for s in SIGMAS:
        g = cv2.GaussianBlur(grey, (0, 0), s)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mean = cv2.GaussianBlur(grey, (0, 0), s)
        sq = cv2.GaussianBlur(grey * grey, (0, 0), s)
        feats += [g, np.sqrt(gx * gx + gy * gy), cv2.Laplacian(g, cv2.CV_32F),
                  np.sqrt(np.maximum(sq - mean * mean, 0)),
                  cv2.GaussianBlur(hsv[..., 1], (0, 0), s)]
    return np.stack(feats, axis=-1)


class PixelForest:
    def __init__(self, n_estimators=100, max_depth=16, samples_per_frame=20000, seed=0):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            n_jobs=-1, random_state=seed)
        self.samples = samples_per_frame
        self.rng = np.random.default_rng(seed)

    def fit(self, frames):
        X, y = [], []
        for f in frames:
            F = pixel_features(f.image).reshape(-1, len(SIGMAS) * 5 + 6)
            m = f.mask.ravel()
            # balanced sample of water and dry pixels
            for cls in (0, 1):
                idx = np.nonzero(m == cls)[0]
                if len(idx):
                    pick = self.rng.choice(idx, min(len(idx), self.samples // 2), replace=False)
                    X.append(F[pick])
                    y.append(m[pick])
        self.model.fit(np.concatenate(X), np.concatenate(y))
        return self

    def predict_proba(self, image):
        F = pixel_features(image)
        p = self.model.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1]
        return p.reshape(image.shape[:2])

    def predict(self, image, threshold=0.5):
        return (self.predict_proba(image) >= threshold).astype(np.uint8)
