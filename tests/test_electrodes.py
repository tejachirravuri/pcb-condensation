import numpy as np

from dropseg.electrodes import find_electrodes
from dropseg.monitor import electrodes_ok


def test_two_combs_found(combs):
    rgb, truth = combs
    el = find_electrodes(rgb)
    assert electrodes_ok(el)
    for k in (1, 2):                    # each found comb is one of the true combs
        vals, counts = np.unique(truth[el.labels == k], return_counts=True)
        assert counts.max() / counts.sum() > 0.95 and vals[np.argmax(counts)] > 0
    assert el.roi.mean() > 0.5
    assert 6 < el.finger_width_px < 14
