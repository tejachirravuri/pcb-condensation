import numpy as np

from dropseg.tracking import link, strongest_growth, track, track_summary

from conftest import disc


def labels_of(*discs, shape=(100, 100)):
    lab = np.zeros(shape, np.int32)
    for i, (y, x, r) in enumerate(discs, 1):
        lab[disc(shape, y, x, r) > 0] = i
    return lab


def test_link_follows_overlap():
    a = labels_of((30, 30, 5), (70, 70, 5))
    b = labels_of((70, 71, 6), (30, 31, 6))          # same droplets, other order
    assert link(a, b) == {1: 2, 2: 1}
    assert link(a, np.zeros_like(a)) == {}


def test_growth_new_merge_vanish():
    seq = [labels_of((30, 30, 4), (30, 50, 4), (80, 80, 3)),
           labels_of((30, 30, 6), (30, 50, 6), (80, 80, 3), (10, 80, 3)),   # grow, one new
           labels_of((30, 40, 13), (10, 80, 4)),                             # merge, one vanished
           labels_of((30, 40, 14), (10, 80, 5))]
    tracks, per = track(seq, interval_min=0.5)
    assert per[1]["new"] == 1 and per[1]["merged"] == 0
    assert abs(per[1]["growth_px_per_min"] - 8.0) < 1.5                    # +4 px diameter in 0.5 min
    assert per[2]["merged"] == 1 and per[2]["absorbed"] == 1
    assert per[2]["vanished"] == 1
    ends = sorted(t.end_reason for t in tracks)
    assert ends.count("merged") == 1 and ends.count("vanished") == 1
    survivor = [t for t in tracks if t.absorbed == 1][0]
    assert len(survivor.areas) == 4 and survivor.label_ids[-1] == 1


def test_summary_and_example():
    seq = [labels_of((50, 50, r)) for r in range(3, 15)]
    tracks, _ = track(seq, interval_min=0.5)
    s = track_summary(tracks, 0.5)
    assert s["long_tracks"] == 1
    assert abs(s["median_growth_px_per_min"] - 4.0) < 0.5                  # +2 px diameter per frame
    assert strongest_growth(tracks).id == tracks[0].id


def test_merge_counted_only_when_confirmed():
    pair = labels_of((30, 30, 5), (30, 50, 5))
    one = labels_of((30, 40, 14))
    _, per = track([pair, pair, one, one])
    assert [p["merged"] for p in per] == [0, 0, 1, 0]
    assert per[2]["absorbed"] == 1
    _, per = track([pair, pair, one, pair])             # flicker: merged for one frame
    assert per[2]["merged"] == 0
