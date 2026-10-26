"""Training and prediction for the U-Net models."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .augment import augment, random_crop
from .models.unet import build


@dataclass
class TrainConfig:
    arch: str = "resnet18"
    base: int = 16                 # plain U-Net only
    depth: int = 4                 # plain U-Net only
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    iters: int = 1200
    batch: int = 8
    crop: int = 256
    dice_weight: float = 1.0
    water_bias: float = 0.6
    seed: int = 0
    threads: int = 0               # 0: torch default


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    union = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def to_tensor(images):
    x = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def train(frames, cfg: TrainConfig = TrainConfig(), log=print, log_every=200):
    """frames: list of data.Frame with image and mask at processing scale."""
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = build(cfg.arch, base=cfg.base, depth=cfg.depth, dropout=cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.iters, pct_start=0.15)
    history = []
    start = time.time()
    model.train()
    for it in range(1, cfg.iters + 1):
        imgs, msks = [], []
        for _ in range(cfg.batch):
            f = frames[int(rng.integers(len(frames)))]
            im, m = random_crop(f.image, f.mask, cfg.crop, rng, cfg.water_bias)
            im, m = augment(im, m, rng)
            imgs.append(im)
            msks.append(m)
        x = to_tensor(imgs)
        y = torch.from_numpy(np.stack(msks).astype(np.float32))[:, None]
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y) + cfg.dice_weight * dice_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == cfg.iters:
            history.append({"iter": it, "loss": float(loss.item()), "s": round(time.time() - start, 1)})
            log("iter %5d  loss %.4f  (%.0fs)" % (it, loss.item(), time.time() - start))
    model.eval()
    return model, history


@torch.no_grad()
def predict_proba(model, image: np.ndarray, tile: Optional[int] = None) -> np.ndarray:
    """Water probability for a full RGB frame. The frame is padded to a
    multiple of the network stride; ``tile`` limits memory on large frames."""
    model.eval()
    H, W = image.shape[:2]
    s = getattr(model, "stride", 32)
    ph, pw = (-H) % s, (-W) % s
    img = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = to_tensor([img])
    prob = torch.sigmoid(model(x))[0, 0].numpy()
    return prob[:H, :W]


def predict(model, image, threshold=0.5):
    return (predict_proba(model, image) >= threshold).astype(np.uint8)


def save(model, cfg: TrainConfig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "config": asdict(cfg)}, path)


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build(cfg.arch, base=cfg.base, depth=cfg.depth, dropout=cfg.dropout,
                  pretrained=False)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, cfg
