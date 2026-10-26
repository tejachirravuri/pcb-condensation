from pathlib import Path

import numpy as np

from dropseg.data import Frame, frame_minute, time_block_folds


def make(minutes):
    z = np.zeros((4, 4), np.uint8)
    return [Frame("f%02d" % i, "A", m, np.zeros((4, 4, 3), np.uint8), z) for i, m in enumerate(minutes)]


def test_frame_minute():
    assert frame_minute(Path("a001.jpg")) == 0.0
    assert frame_minute(Path("rec/a020.jpg")) == 9.5


def test_time_blocks_keep_buffer():
    frames = make([2.5 * i for i in range(15)])
    by_id = {f.id: f for f in frames}
    seen = []
    for train, test in time_block_folds(frames, n_blocks=5, buffer_min=2.5):
        seen += test
        lo = min(by_id[i].minute for i in test)
        hi = max(by_id[i].minute for i in test)
        for i in train:
            m = by_id[i].minute
            assert m < lo - 2.5 or m > hi + 2.5
    assert sorted(seen) == sorted(by_id)
