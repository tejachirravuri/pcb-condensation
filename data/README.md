# Data

Internal Bosch test data from 2021/22. Not for redistribution.

```
data/
  labelled/
    images/<id>.png        21 labelled frames, original colours
    masks/<id>.png         water = 255
    instances/<id>.json    droplet polygons (only 6 frames have them)
    index.csv              id, session, minute, size, water fraction, label source
  raw_labels/              the label files as exported from the tools
    labelme/               Labelme polygons, recording a
    apeer_masks/           Apeer brush masks, recording a (dark = water)
    reference_masks/       masks of the reference measurement (bright = water)
  recordings/a ... j/      1,212 frames from ten condensation phases, 1280x960,
                           one frame every 30 s (a001 is the first frame)
```

`scripts/build_dataset.py --source <project folder> --layout <layout.json>` makes
`labelled/` and `recordings/` from the original folders.

## Labelled frames

| session | frames | size | labels |
|---|---|---|---|
| A (recording a) | a20, a25, a30, a35 | 1280x960 | Labelme polygons per droplet, size class mini / small / medium |
| A | a43 | 1280x960 | Labelme, only part of the frame labelled, **not used** |
| A | a45 to a90, every 5th frame | 1280x960 | Apeer brush masks; a65 also as polygons (57 droplets) |
| R (reference measurement, Feb 2022) | a3, a4, a8, a12, a14, a21 | 512x512 crops | Apeer masks, different camera position and zoom |

Frame `aNN` of recording a was taken (NN - 1) x 30 s after the start of the
condensation phase. Session R comes from a separate video with its own
numbering.

Water covers 21 % to 66 % of the frame in session A and 2 % to 29 % in
session R, so R is the closer one to the early stage of condensation.

Label quality differs between the tools: Labelme polygons outline single
droplets and leave thin dry gaps between them; Apeer masks are painted with
a brush and merge neighbouring droplets. A model trained on one style is
scored against the other in some folds, which lowers the scores somewhat.

A file of model predictions that was lying next to the labels (`c13`) is
not used as ground truth.
