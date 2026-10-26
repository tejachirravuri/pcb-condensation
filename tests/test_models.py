import numpy as np
import torch

from dropseg.data import Frame
from dropseg.models.unet import build
from dropseg.train import TrainConfig, load, predict_proba, save, train

from conftest import comb_image, disc


def frames(n=2):
    out = []
    for k in range(n):
        rgb, _ = comb_image(96, 128)
        m = disc((96, 128), 40 + 5 * k, 60, 12)
        rgb = rgb.copy()
        rgb[m > 0] = (120, 140, 170)
        out.append(Frame("s%d" % k, "A", float(k), rgb, m))
    return out


def test_output_shapes():
    for arch, kw in (("plain", {"base": 8}), ("resnet18", {"pretrained": False})):
        net = build(arch, **kw).eval()
        with torch.no_grad():
            y = net(torch.zeros(1, 3, 64, 96))
        assert y.shape == (1, 1, 64, 96)


def test_train_predict_save_load(tmp_path):
    cfg = TrainConfig(arch="plain", base=8, iters=3, batch=2, crop=64, threads=1)
    model, hist = train(frames(), cfg, log=lambda *a: None)
    assert hist and np.isfinite(hist[-1]["loss"])
    img = frames(1)[0].image[:77, :101]          # not a multiple of the stride
    p = predict_proba(model, img)
    assert p.shape == (77, 101) and 0 <= p.min() and p.max() <= 1
    save(model, cfg, tmp_path / "m.pt")
    model2, cfg2 = load(tmp_path / "m.pt")
    assert cfg2.arch == "plain"
    assert np.allclose(predict_proba(model2, img), p, atol=1e-5)


def test_maskrcnn_targets():
    from dropseg.models.maskrcnn import target_from_labels
    lab = np.zeros((50, 50), np.int32)
    lab[5:10, 5:10] = 1
    lab[20:30, 30:45] = 2
    lab[0, 0] = 3                                 # too small
    t = target_from_labels(lab, min_area=6)
    assert t["boxes"].shape == (2, 4)
    assert t["boxes"][1].tolist() == [30, 20, 45, 30]
    assert t["masks"].shape == (2, 50, 50)
    assert target_from_labels(np.zeros((5, 5), np.int32), 1) is None
