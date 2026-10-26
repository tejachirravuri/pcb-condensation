"""Mask R-CNN for single droplets.

torchvision's Mask R-CNN (ResNet-50 FPN) started from COCO weights, with
one foreground class. Training runs on a CPU, so the lighter first version
of the heads is the default and fewer region proposals are kept. The droplets are much smaller than COCO objects (a
few to about 80 px across in the full 1280x960 frame), so the anchors are
four times smaller than the default. The image is never rescaled inside
the model: training uses full resolution crops and prediction the full frame.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch

from ..metrics import rasterise

ANCHORS = ((8,), (16,), (32,), (64,), (128,))


def build(pretrained=True, size=512, max_size=1333, detections=400, version=1):
    from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = "DEFAULT" if pretrained else None
    make = maskrcnn_resnet50_fpn_v2 if version == 2 else maskrcnn_resnet50_fpn
    model = make(weights=weights, weights_backbone=None, min_size=size, max_size=max_size,
                 box_detections_per_img=detections,
                 rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=500,
                 rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=1000,
                 box_batch_size_per_image=256)
    model.rpn.anchor_generator = AnchorGenerator(ANCHORS, ((0.5, 1.0, 2.0),) * len(ANCHORS))
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    return model


def set_input_size(model, height, width):
    """Make the internal transform keep a frame of this size as it is."""
    model.transform.min_size = (min(height, width),)
    model.transform.max_size = max(height, width)


@dataclass
class MRCNNConfig:
    version: int = 1
    iters: int = 600
    batch: int = 2
    crop: int = 320
    lr: float = 0.01
    weight_decay: float = 1e-4
    min_area: int = 6           # pixels of a droplet that must be inside a crop
    seed: int = 0
    threads: int = 0


def target_from_labels(lab: np.ndarray, min_area: int):
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    ids = ids[counts >= min_area]
    if not len(ids):
        return None
    masks = np.stack([(lab == i) for i in ids]).astype(np.uint8)
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(ids), dtype=torch.int64),
            "masks": torch.from_numpy(masks)}


def _sample(frame, labels, cfg, rng):
    H, W = labels.shape
    s = cfg.crop
    for _ in range(10):
        y0 = int(rng.integers(0, max(H - s, 0) + 1))
        x0 = int(rng.integers(0, max(W - s, 0) + 1))
        img = frame.image[y0:y0 + s, x0:x0 + s]
        lab = labels[y0:y0 + s, x0:x0 + s]
        if rng.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        if rng.random() < 0.5:
            img, lab = img[::-1], lab[::-1]
        k = int(rng.integers(4))
        img, lab = np.ascontiguousarray(np.rot90(img, k)), np.ascontiguousarray(np.rot90(lab, k))
        im = img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
        target = target_from_labels(lab, cfg.min_area)
        if target is not None:
            x = torch.from_numpy(np.clip(im, 0, 255) / 255.0).permute(2, 0, 1).float()
            return x, target
    return None


def train(frames, cfg: MRCNNConfig = MRCNNConfig(), log=print, log_every=100):
    """frames: data.Frame objects at full resolution with instance polygons."""
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    labels = [rasterise(f.instances, f.mask.shape) for f in frames]
    model = build(size=cfg.crop, version=cfg.version)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.iters, pct_start=0.1)
    model.train()
    history = []
    for it in range(1, cfg.iters + 1):
        xs, ts = [], []
        while len(xs) < cfg.batch:
            k = int(rng.integers(len(frames)))
            s = _sample(frames[k], labels[k], cfg, rng)
            if s is not None:
                xs.append(s[0])
                ts.append(s[1])
        losses = model(xs, ts)
        loss = sum(losses.values())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == cfg.iters:
            row = {"iter": it, "loss": float(loss.item())}
            row.update({k: round(float(v.item()), 4) for k, v in losses.items()})
            history.append(row)
            log("iter %4d  loss %.3f" % (it, loss.item()))
    model.eval()
    return model, history


@torch.no_grad()
def predict(model, image: np.ndarray, score_thr=0.5, mask_thr=0.5):
    """Label image of droplets (higher scoring detections drawn last) and scores."""
    model.eval()
    H, W = image.shape[:2]
    set_input_size(model, H, W)
    x = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    out = model([x])[0]
    keep = out["scores"] >= score_thr
    masks = (out["masks"][keep, 0] >= mask_thr).numpy()
    scores = out["scores"][keep].numpy()
    lab = np.zeros((H, W), np.int32)
    for i in np.argsort(scores):
        lab[masks[i]] = i + 1
    # renumber so ids are 1..n
    ids = np.unique(lab[lab > 0])
    lut = np.zeros(len(scores) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab], scores


def save(model, cfg: MRCNNConfig, path):
    """Weights are stored in half precision, which halves the file (about
    85 MB) and makes no visible difference to the masks."""
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"state": state, "config": asdict(cfg)}, path)


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MRCNNConfig(**ckpt["config"])
    model = build(pretrained=False, version=cfg.version)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state"].items()})
    model.eval()
    return model, cfg
