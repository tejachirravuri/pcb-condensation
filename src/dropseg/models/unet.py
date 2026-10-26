"""U-Net for water/dry segmentation.

Two encoders:
- ``plain``: the original U-Net layout (two 3x3 convs per level), trained
  from scratch, width and depth configurable.
- ``resnet18``: torchvision ResNet-18 with ImageNet weights as encoder and a
  U-Net decoder. With only about twenty labelled frames, starting from
  pretrained features matters.
Both output one logit per pixel (water).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout, dropout=0.0):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class Up(nn.Module):
    def __init__(self, cin, cskip, cout, dropout=0.0):
        super().__init__()
        self.block = conv_block(cin + cskip, cout, dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class PlainUNet(nn.Module):
    def __init__(self, base=16, depth=4, dropout=0.0):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        cin = 3
        for c in chans[:-1]:
            self.down.append(conv_block(cin, c))
            cin = c
        self.bottom = conv_block(chans[-2], chans[-1], dropout)
        self.up = nn.ModuleList(Up(chans[i + 1], chans[i], chans[i], dropout if i == depth - 1 else 0.0)
                                for i in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], 1, 1)
        self.stride = 2 ** depth

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for up, skip in zip(self.up, reversed(skips)):
            x = up(x, skip)
        return self.head(x)


class ResNetUNet(nn.Module):
    """ResNet-18 encoder (strides 2..32) with a U-Net decoder."""

    def __init__(self, pretrained=True, dropout=0.0):
        super().__init__()
        import torchvision
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)           # /2, 64
        self.pool = r.maxpool                                       # /4
        self.enc = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])  # /4 64, /8 128, /16 256, /32 512
        self.up4 = Up(512, 256, 256, dropout)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up1 = Up(64, 64, 32)
        self.up0 = Up(32, 3, 16)
        self.head = nn.Conv2d(16, 1, 1)
        self.stride = 32
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        raw = x
        x = (x - self.mean) / self.std
        s1 = self.stem(x)
        e1 = self.enc[0](self.pool(s1))
        e2 = self.enc[1](e1)
        e3 = self.enc[2](e2)
        e4 = self.enc[3](e3)
        d = self.up4(e4, e3)
        d = self.up3(d, e2)
        d = self.up2(d, e1)
        d = self.up1(d, s1)
        d = self.up0(d, raw)
        return self.head(d)


def build(arch="resnet18", **kw):
    if arch == "plain":
        return PlainUNet(base=kw.get("base", 16), depth=kw.get("depth", 4), dropout=kw.get("dropout", 0.0))
    if arch == "resnet18":
        return ResNetUNet(pretrained=kw.get("pretrained", True), dropout=kw.get("dropout", 0.0))
    raise ValueError("unknown architecture %r" % arch)
