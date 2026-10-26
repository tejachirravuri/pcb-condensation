"""Labelled frames, recordings and cross-validation folds."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# a043 was labelled with a different class scheme and only covers part of the
# water (10 % against 51 % and 56 % one minute before and after)
EXCLUDED = {"A_a043": "incomplete labels"}

# processing scale per session: session A frames are 1280x960, the
# reference crops are already 512 px
SCALE = {"A": 0.5, "R": 1.0}


@dataclass
class Frame:
    id: str
    session: str
    minute: Optional[float]
    image: np.ndarray            # RGB uint8, at processing scale
    mask: np.ndarray             # 0/1 uint8, at processing scale
    instances: List[np.ndarray] = field(default_factory=list)  # polygons, processing scale
    instance_classes: List[str] = field(default_factory=list)
    label_source: str = ""

    @property
    def has_instances(self) -> bool:
        return bool(self.instances)


def read_index(root: Path = DATA) -> List[dict]:
    with (root / "labelled" / "index.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def load_frame(row: dict, root: Path = DATA, scale: Optional[float] = None) -> Frame:
    lab = root / "labelled"
    scale = SCALE[row["session"]] if scale is None else scale
    image = cv2.cvtColor(cv2.imread(str(lab / "images" / ("%s.png" % row["id"]))), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(lab / "masks" / ("%s.png" % row["id"])), 0)
    if scale != 1.0:
        size = (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    polys, classes = [], []
    inst = lab / "instances" / ("%s.json" % row["id"])
    if inst.is_file():
        for d in json.loads(inst.read_text())["droplets"]:
            polys.append(np.asarray(d["points"], np.float32) * scale)
            classes.append(d["size_class"])
    minute = float(row["minute"]) if row.get("minute") not in (None, "") else None
    return Frame(row["id"], row["session"], minute, image, (mask > 0).astype(np.uint8),
                 polys, classes, row.get("label_source", ""))


def load_frames(sessions=("A", "R"), include_excluded=False, root: Path = DATA) -> List[Frame]:
    return [load_frame(r, root) for r in read_index(root)
            if r["session"] in sessions and (include_excluded or r["id"] not in EXCLUDED)]


def time_block_folds(frames: List[Frame], n_blocks: int = 5, buffer_min: float = 2.5):
    """Folds of consecutive frames for one recording.

    Neighbouring frames are only 2.5 minutes apart and look much alike, so
    frames within ``buffer_min`` of any test frame are also removed from the
    training set of that fold. Yields (train_ids, test_ids)."""
    timed = sorted((f for f in frames if f.minute is not None), key=lambda f: f.minute)
    blocks = np.array_split(np.arange(len(timed)), n_blocks)
    for block in blocks:
        test = [timed[i] for i in block]
        lo, hi = test[0].minute - buffer_min - 1e-6, test[-1].minute + buffer_min + 1e-6
        train = [f for f in timed if not lo <= f.minute <= hi]
        yield [f.id for f in train], [f.id for f in test]


def recording_frames(name: str, root: Path = DATA) -> List[Path]:
    files = sorted((root / "recordings" / name).glob("*.jpg"),
                   key=lambda p: int(re.findall(r"(\d+)", p.stem)[0]))
    return files


def recordings(root: Path = DATA) -> List[str]:
    return sorted(p.name for p in (root / "recordings").iterdir() if p.is_dir())


def frame_minute(path: Path, interval_min: float = 0.5) -> float:
    return (int(re.findall(r"(\d+)", Path(path).stem)[0]) - 1) * interval_min
