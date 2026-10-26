"""Following single droplets from frame to frame.

Frames are 30 s apart and droplets on the board hardly move in that time,
so a droplet in the next frame is linked to the droplets of the previous
frame it covers. A droplet of frame t goes to the droplet of frame t+1 that
covers most of it, if that is at least ``min_cover`` of its area. Then, for
every droplet of frame t+1:

    no droplet came in    new        (nucleated, or grew big enough to be seen)
    one droplet came in   grown      (same droplet, area may have changed)
    two or more came in   merged     (coalescence)

and a droplet of frame t that went nowhere has vanished (evaporated, or
lost by the segmentation). The segmentation flickers a little from frame
to frame, so a new droplet is only counted if it is still there in the next
frame, and a merge only if the merged droplet is still there in the next
frame and the droplets that went into it had been seen for two frames. A droplet keeps its track id while it grows; at
a merge the new droplet carries on the track of the largest one that went
into it, and the other tracks end there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Track:
    id: int
    start: int                                   # frame index
    areas: List[float] = field(default_factory=list)
    label_ids: List[int] = field(default_factory=list)   # droplet id in each frame
    end_reason: str = "end"                      # end, merged, vanished
    absorbed: int = 0                            # droplets merged into this one

    @property
    def end(self) -> int:
        return self.start + len(self.areas) - 1

    def diameters(self):
        return 2 * np.sqrt(np.asarray(self.areas) / np.pi)


def link(prev: np.ndarray, cur: np.ndarray, min_cover: float = 0.3):
    """Map droplet ids of ``prev`` to ids of ``cur``. Returns {prev_id: cur_id}."""
    P, C = int(prev.max()), int(cur.max())
    if not P or not C:
        return {}
    pair = prev.astype(np.int64) * (C + 1) + cur
    ov = np.bincount(pair.ravel(), minlength=(P + 1) * (C + 1)).reshape(P + 1, C + 1)
    area = ov.sum(1)
    best = ov[1:, 1:].argmax(1) + 1
    cover = ov[np.arange(1, P + 1), best] / np.maximum(area[1:], 1)
    return {int(i + 1): int(best[i]) for i in range(P) if area[i + 1] and cover[i] >= min_cover}


def track(labels: Sequence[np.ndarray], scale_to_full: float = 1.0, interval_min: float = 0.5,
          min_cover: float = 0.3):
    """Tracks over a sequence of droplet label images and per frame counts.

    Returns (tracks, per_frame) with per_frame[t] = {"new", "merged",
    "absorbed", "vanished", "growth_px_per_min"}: droplets that appeared,
    merges, droplets that went into a merge (these three confirmed as
    described above), droplets that disappeared, and the median change of
    diameter of the droplets that only grew. Areas and diameters are in full
    frame pixels."""
    a2 = scale_to_full ** 2
    tracks: Dict[int, Track] = {}
    active: Dict[int, int] = {}                  # droplet id in current frame -> track id
    per_frame, merges = [], []
    for t, lab in enumerate(labels):
        ids = np.unique(lab[lab > 0])
        areas = dict(zip(ids.tolist(), (np.bincount(lab.ravel())[ids] * a2).tolist()))
        stats = {"new": 0, "merged": 0, "absorbed": 0, "vanished": 0, "growth_px_per_min": float("nan")}
        new_active = {}
        if t == 0:
            for d in ids.tolist():
                tr = Track(len(tracks) + 1, 0, [areas[d]], [d])
                tracks[tr.id] = tr
                new_active[d] = tr.id
        else:
            mapping = link(labels[t - 1], lab, min_cover)
            incoming: Dict[int, List[int]] = {}
            for p, c in mapping.items():
                incoming.setdefault(c, []).append(p)
            stats["vanished"] = sum(1 for p in active if p not in mapping)
            for p in active:
                if p not in mapping:
                    tracks[active[p]].end_reason = "vanished"
            growth = []
            for d in ids.tolist():
                parents = [p for p in incoming.get(d, []) if p in active]
                if not parents:
                    tr = Track(len(tracks) + 1, t, [areas[d]], [d])
                    tracks[tr.id] = tr
                else:
                    parents.sort(key=lambda p: -tracks[active[p]].areas[-1])
                    tr = tracks[active[parents[0]]]
                    if len(parents) == 1:
                        d0 = 2 * np.sqrt(tr.areas[-1] / np.pi)
                        growth.append(2 * np.sqrt(areas[d] / np.pi) - d0)
                    else:
                        merges.append((t, tr.id, [active[p] for p in parents]))
                        tr.absorbed += len(parents) - 1
                        for p in parents[1:]:
                            tracks[active[p]].end_reason = "merged"
                    tr.areas.append(areas[d])
                    tr.label_ids.append(d)
                new_active[d] = tr.id
            if growth:
                stats["growth_px_per_min"] = float(np.median(growth)) / interval_min
        active = new_active
        per_frame.append(stats)
    for tr in tracks.values():
        if tr.start > 0 and len(tr.areas) >= 2:
            per_frame[tr.start]["new"] += 1
    for t, survivor, went_in in merges:
        if tracks[survivor].end > t and all(tracks[i].start <= t - 2 for i in went_in):
            per_frame[t]["merged"] += 1
            per_frame[t]["absorbed"] += len(went_in) - 1
    return list(tracks.values()), per_frame


def track_summary(tracks: Sequence[Track], interval_min: float, min_frames: int = 6) -> dict:
    """Growth of droplets that were followed for at least ``min_frames`` frames."""
    long = [tr for tr in tracks if len(tr.areas) >= min_frames]
    if not long:
        return {"tracks": len(tracks), "long_tracks": 0}
    rates = []
    for tr in long:
        d = tr.diameters()
        t = np.arange(len(d)) * interval_min
        rates.append(np.polyfit(t, d, 1)[0])
    ends = [tr.end_reason for tr in tracks]
    return {"tracks": len(tracks), "long_tracks": len(long),
            "median_growth_px_per_min": float(np.median(rates)),
            "p90_growth_px_per_min": float(np.percentile(rates, 90)),
            "ended_by_merge": ends.count("merged"), "ended_by_vanish": ends.count("vanished"),
            "most_absorbed": int(max(tr.absorbed for tr in tracks))}


def strongest_growth(tracks: Sequence[Track], min_frames: int = 10, max_diameter: float = 200,
                     accept=None):
    """Track with the largest increase in diameter, for showing an example.
    Regions larger than ``max_diameter`` are water films, not droplets;
    ``accept(track)`` can rule out more."""
    cands = [tr for tr in tracks if len(tr.areas) >= min_frames and tr.diameters().max() <= max_diameter
             and (accept is None or accept(tr))]
    if not cands:
        return None
    return max(cands, key=lambda tr: tr.diameters()[-1] - tr.diameters()[0])
