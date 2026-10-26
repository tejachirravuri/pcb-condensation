import numpy as np

from dropseg.droplets import droplet_table, split_droplets, summary

from conftest import disc


def test_touching_droplets_are_split():
    w = disc((80, 120), 40, 40, 15) | disc((80, 120), 40, 68, 15)
    assert len(np.unique(w)) == 2
    lab = split_droplets(w)
    assert len(np.unique(lab[lab > 0])) == 2
    assert (lab > 0).sum() == w.sum()          # no water is lost


def test_separate_and_tiny_droplets():
    w = disc((60, 60), 15, 15, 6) | disc((60, 60), 45, 45, 8)
    w[2, 50] = 1                                # a single pixel speck
    lab = split_droplets(w, min_area_px=4)
    assert len(np.unique(lab[lab > 0])) == 2


def test_table_diameter_and_scale():
    lab = split_droplets(disc((100, 100), 50, 50, 20))
    rows = droplet_table(lab, scale_to_full=2.0)
    assert len(rows) == 1
    assert abs(rows[0]["diameter_px"] - 80) < 3     # 2 x 2r at full size
    assert abs(rows[0]["x"] - 100) < 1


def test_summary():
    w = disc((100, 100), 20, 20, 5) | disc((100, 100), 20, 60, 5) | disc((100, 100), 70, 40, 10)
    s = summary(droplet_table(split_droplets(w)), roi_area_px=1e4)
    assert s["count"] == 3
    assert s["density_per_mpx"] == 300
    assert s["max_diameter_px"] > s["median_diameter_px"]
    assert summary([], 1e4)["count"] == 0


def test_slivers_removed_round_droplets_kept():
    from dropseg.droplets import remove_slivers
    lab = np.zeros((60, 120), np.int32)
    lab[10:13, 5:100] = 1                       # thin line along a finger edge
    lab[disc((60, 120), 40, 30, 6) > 0] = 2     # small round droplet
    lab[30:50, 60:110] = 3                      # wide, somewhat long water area
    out = remove_slivers(lab, min_width_px=8)
    assert set(np.unique(out)) == {0, 2, 3}
