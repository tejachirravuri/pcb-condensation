# Method

## Frames and scale

Recording a is 1280x960. The networks work on frames scaled to half size
(640x480): a droplet of 10 px at full size is still 5 px wide, and training
on a laptop CPU becomes about four times faster. The reference frames
(session R) are 512x512 crops at a higher zoom and are used as they are.
Mask R-CNN works on the full 1280x960 frame, since its smallest feature map
has a stride of 4.

## Validation

With 20 usable labelled frames, how the frames are split decides what the
score means.

- **Time-block cross-validation on recording a.** The 14 frames are put in
  time order and cut into 5 blocks of 2 to 3 consecutive frames. Each block
  is the test set once. Frames within 2.5 min of a test frame are also
  removed from the training set of that fold, because neighbouring frames
  are nearly identical.
- **Other setup.** Train on all of recording a and test on the six frames of
  the reference measurement (different camera position, zoom and day).
  Then add half of the reference frames to training and test on the other
  half, both ways round.

Scores are Dice, IoU, precision and recall of the water class, per frame and
pooled over all test pixels. The probability threshold is 0.5 everywhere;
nothing is tuned on the test frames.

## Models for water / dry

**Pixel forest.** A random forest (100 trees, depth 16) on 26 features per
pixel: RGB, HSV, and for four Gaussian scales the blurred grey value,
gradient magnitude, Laplacian, local standard deviation and blurred
saturation. 20,000 pixels per frame, balanced between water and dry. This
is the kind of classifier interactive tools like ilastik train, and it is
the baseline.

**Plain U-Net.** Four down-sampling levels, 16 to 256 channels, two 3x3
convolutions with batch norm per level, bilinear up-sampling, 1.96 M
parameters, trained from scratch.

**ResNet-18 U-Net.** The encoder is torchvision's ResNet-18 with ImageNet
weights, the decoder is the same as above with a skip from the input
resolution as well. 14.3 M parameters.

Both U-Nets are trained the same way: random 192x192 crops (60 % of them
centred near a water pixel, so sparse droplets are seen often enough),
batch 4, 800 steps, AdamW with a one-cycle learning rate schedule, loss =
binary cross-entropy + Dice. Augmentation is applied on the fly to the
training crops only: flips, 90 degree rotations, small rotation and zoom,
brightness, contrast, gamma, colour cast, blur and noise.

## Single droplets

**Watershed.** A water mask is split into droplets on the distance
transform: every local maximum of the (slightly smoothed) distance to the
dry area, at least 10 px from the next one at full size, seeds a droplet,
and the seeds are flooded with an image foresting transform. Dry pixels are
the highest ground, so a flood never crosses them. Regions without any seed
get one at their deepest point. Specks below 12 px (full size) are dropped.

**Mask R-CNN.** torchvision's Mask R-CNN with a ResNet-50 FPN backbone,
started from COCO weights, one droplet class. The anchors are four times
smaller than the default (8 to 128 px) because the droplets are far smaller
than COCO objects. It is trained on 320x320 crops of the full size frames
with flips and rotations, SGD with a one-cycle schedule, 400 steps. Only
five frames have every droplet outlined (a20, a25, a30, a35, a65), so it is
evaluated in three folds over these frames with the neighbouring frames
left out of training.

Droplets are matched to the labelled droplets one to one by IoU >= 0.5
(greedy, highest IoU first). A labelled droplet counts as *seen* if at
least half of its pixels are predicted as water; this gives the recall per
size class and the smallest droplet that is reliably seen.

**Thin lines are not droplets.** On a dry board the network marks some of
the thin bright and dark lines along the edges of the comb fingers as
water. Droplets are round, so a region is dropped if it is more than three
times longer than wide and narrower than 16 px (full size), or more than
ten times longer than wide at any width. On the outlined frames this
removes 2 of 582 hand-drawn droplets; on dry frames it removes more than
half of the false detections.

For every droplet: area, equivalent diameter, centroid and distance to the
nearest neighbour. Per frame: count, density per megapixel of the comb
area, median, 10th and 90th percentile diameter, and the wet share of the
comb area (area of the droplets kept).

## Comb area

Coverage and droplets are measured inside the comb structure only, not on
the frame around it. The camera does not move during a recording, so the
structure is found once, on the driest frame: metal on these boards is
bright and bluish while the solder mask is yellow, so metal = (B - R > 15)
and max(R, G, B) > 110, cleaned with a small opening and closing. The two
largest connected metal regions are the two combs (checked: similar size,
each more than 3 % of the frame), and the comb area is the convex hull of
both. The median finger width (about 30 px at full size) is a convenient
ruler for droplet sizes, since the frames have no scale bar.

## Following droplets over time

Frames are 30 s apart and droplets hardly move in that time, so each
droplet of a frame is linked to the droplet of the next frame that covers
most of it (at least 30 % of its area). For every droplet of the next
frame this gives one of:

- **new**: no droplet came in. It has just nucleated or has just grown big
  enough to be seen.
- **grown**: one droplet came in. Same droplet; the change of its diameter is
  its growth.
- **merged**: two or more came in (coalescence). The merged droplet carries on
  the track of the largest one; the other tracks end there.

A droplet that goes nowhere has **vanished**: it evaporated, or the
segmentation lost it. The segmentation flickers a little from frame to
frame, and a watershed can split a droplet in one frame and not in the
next. So a new droplet only counts if it is still there one frame later,
and a merge only counts if the merged droplet is still there one frame
later and the droplets that went into it had been seen for at least two
frames.

Per frame this gives the number of droplets, their size distribution
(median, 10th and 90th percentile diameter), the number of new droplets and
merges, and the median growth of the droplets that only grew. Per droplet
it gives a track: when it appeared, how its diameter changed, how many
droplets it swallowed and how it ended. The growth rate of a droplet is the
slope of a straight line through its diameter over time, for droplets
followed for at least 3 minutes.

## Events in a recording

Each recording was cut from the chamber video around one condensation
phase, so all of them start with some water already on the board. The
events describe the phases inside a recording:

    peak count     the frame with the most droplets before the wettest
                   frame: nucleation is over
    peak coverage  the wettest frame
    merge phase    from peak count to peak coverage
    dry again      first frame after the peak with less than a quarter of
                   the peak coverage

In the merge phase the number of droplets falls while the wet area still
grows: droplets grow by swallowing their neighbours, and wide water areas
form between the fingers. It is described by how far the count drops and
how far the 90th percentile diameter rises between its two ends, and by the
merges counted on the way. Counts and coverage are smoothed with a running
median over five frames (2.5 min) before the peaks are picked. A recording
that is wettest at its start has no merge phase; it only dries.

## Hyper-parameter search

Optuna (TPE sampler, 10 trials of 400 steps) over architecture, learning
rate, Dice weight, crop size, water bias of the crops and dropout. The
search uses its own split of recording a (training up to a55, validation
a70 to a90, a60 and a65 left out as buffer) and its first trial is the
setting used in the cross-validation. The best setting is used only for the
final model; the cross-validation numbers were produced before the search.

## Final model

The final U-Net is trained on all 20 frames for 1,500 steps and used for
the recordings and the command line tool.

## Droplet growth in the hand labels

`scripts/plot_label_growth.py` measures the droplets in the five frames
that are outlined by hand. No model is involved, so this is the reference
for what the tracking should find in recording a.
