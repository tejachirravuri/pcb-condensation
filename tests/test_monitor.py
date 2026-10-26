import cv2
import numpy as np

from dropseg.monitor import analyse, events, first_confirmed, growth_strip, overlay

from conftest import comb_image, disc


def test_first_confirmed_needs_two_frames():
    assert first_confirmed([False, True, False, True, True]) == 3
    assert first_confirmed([True, False]) is None


def row(minute, coverage, count, merged):
    return {"minute": minute, "coverage": coverage, "count": count, "merged": merged,
            "median_diameter_px": 10.0, "p90_diameter_px": 20.0}


def test_events():
    # droplets appear (count up), merge (count down, coverage up), then dry
    count = [0, 20, 60, 100, 120, 110, 90, 70, 50, 40, 35, 30, 20, 5, 0, 0]
    cover = [0, .02, .05, .1, .15, .2, .25, .3, .35, .4, .42, .4, .2, .05, 0, 0]
    rows = [{"minute": 0.5 * i, "coverage": c, "count": n, "merged": 2, "new": 1,
             "p90_diameter_px": 20 + 10 * i} for i, (n, c) in enumerate(zip(count, cover))]
    ev = events(rows)
    assert ev["onset_min"] == 0.5
    # five-frame running median: peaks may move by a frame
    assert ev["peak_count_min"] in (1.5, 2.0)
    assert ev["peak_coverage_min"] in (4.5, 5.0)
    assert ev["merge_phase_min"] >= 2.5
    assert ev["count_ratio"] < 0.5 and ev["p90_ratio"] > 1.5
    assert ev["dry_again_min"] == 6.5
    assert not ev["starts_wet"]


def test_drying_only_has_no_merge_phase():
    rows = [{"minute": 0.5 * i, "coverage": 0.5 - 0.05 * i, "count": 10 + i, "p90_diameter_px": 50.0}
            for i in range(10)]
    ev = events(rows)
    assert ev["starts_wet"] and ev["peak_coverage_min"] == 0.0
    assert "count_ratio" not in ev


def sequence(tmp_path):
    """Two droplets that grow, touch and merge, between the fingers of a comb."""
    rgb, truth = comb_image(240, 320)
    cx = int(np.nonzero(truth[120] == 1)[0][0]) + 5          # on the first finger
    radii = [0, 0, 3, 4, 5, 6, 7, -11, -12]                     # merged into one at the end
    paths = []
    for i, r in enumerate(radii, 1):
        im = rgb.copy()
        if r > 0:
            m = disc(truth.shape, 100, cx, r) | disc(truth.shape, 100 + 17, cx, r)
            im[m > 0] = (255, 255, 255)                         # water drawn white
        elif r < 0:
            im[disc(truth.shape, 108, cx, -r) > 0] = (255, 255, 255)
        p = tmp_path / ("x%03d.png" % i)
        cv2.imwrite(str(p), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        paths.append(p)
    return paths


def predict(im):                                                 # stand-in for the network
    return (im.min(axis=2) > 240).astype(np.float32)


def test_analyse_sequence(tmp_path):
    res = analyse(sequence(tmp_path), predict, render=True, scale=1.0)
    rows, ev = res["rows"], res["events"]
    assert ev["electrodes_found"]
    assert [r["count"] for r in rows] == [0, 0, 2, 2, 2, 2, 2, 1, 1]
    assert [r["new"] for r in rows][:3] == [0, 0, 2]
    assert rows[7]["merged"] == 1 and ev["total_merges"] == 1
    assert rows[5]["growth_px_per_min"] > 0
    assert len(res["rendered"]) == len(rows)
    tr = max(res["tracks"], key=lambda t: len(t.areas))
    strip = growth_strip(res["images"], res["labels"], tr)
    assert strip.ndim == 3 and strip.shape[1] > strip.shape[0]


def test_overlay_shapes():
    rgb, _ = comb_image(60, 80)
    w = disc((60, 80), 30, 40, 8)
    lab = w.astype(np.int32)
    assert overlay(rgb, w, lab, scale=1.0).shape == (60, 80, 3)
