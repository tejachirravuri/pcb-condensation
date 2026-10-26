# Background

I worked on this during my internship at Bosch (Automotive Electronics,
Engineering Assembly and Interconnect Technology, Schwieberdingen) in the
winter of 2021/22. This repository is a rewrite of that work from scratch,
done in 2026 with the same recordings and labels.

## The problem

Control units are qualified in climate cycling chambers: temperature and
humidity are cycled until water condenses on the boards. Condensation,
ionic contamination and a voltage between neighbouring conductors together
cause electrochemical migration (ECM). Metal ions move through the water
film, dendrites grow from one conductor to the other, and the result is a
leakage path or a short. On an ECU in a car that is a field failure.

The test boards carry interdigitated comb structures: two comb-shaped
electrodes whose fingers interlock with a narrow gap between them.
A camera with a vibration damper looks at one comb structure inside the
chamber. The question from the department was what the camera can tell us:

- how much of the structure is covered with water, frame by frame,
- the size and number of droplets, and the smallest droplet that can still
  be seen at the start of condensation (earlier lab work found that the
  small droplets between the fingers matter more for ECM than the big ones),
- whether the method is fast enough to run next to a test.

The images are hard. The board is bright and shiny, there is no natural
light in the chamber, and a thin water film over the whole surface looks
almost the same as a dry surface. Droplet edges are often the only thing
that is visible.

## What was done in 2021/22

An earlier student project had trained a U-Net on a few frames and reached
a Dice score of about 0.74. In my part I extracted frames from the chamber
videos (one frame every 30 s within each condensation phase), labelled
water by hand in Labelme (polygons per droplet, with a size class) and in
Apeer (brush masks), built a dataset of 500 images by augmenting 10
labelled frames, and trained U-Net variants on it. The best one reached a
Dice of 0.90 on the validation split, and Attention U-Net 0.87.

Two things were wrong with that number, and I only understood them later:

1. The 500 images came from 10 frames and were split into training and
   validation at random afterwards. Augmented copies of the same frame
   were on both sides of the split, so the validation set measured how well
   the network remembered 10 frames, not how it does on a new one.
2. The augmented crops were saved as TIFF with red and blue swapped,
   so the network was trained on colours it never sees on a real frame.
   This is why predictions on augmented images looked good and predictions
   on the original frames did not.

There was also no droplet level analysis (counting, sizes) and no
connection to the actual failure mechanism.

## What this rewrite does differently

- All 21 labelled frames are used, in their original colours, and every
  score comes from frames the model has not seen during training. Frames
  close in time to a test frame are also held back (time-block
  cross-validation), since two frames 30 s apart are nearly the same image.
- The 2022 protocol is reproduced on purpose (`scripts/run_leaky_protocol.py`)
  to show how far the random split overstates the result.
- A pixel classifier (random forest), a plain U-Net and a U-Net with a
  pretrained ResNet-18 encoder are compared, and Mask R-CNN is tried for
  single droplets.
- The focus is on the droplets themselves: how they form, grow and merge.
  Every droplet is segmented, measured and followed from frame to frame,
  so a recording becomes a record of new droplets, growth rates and merges
  instead of a single "percent wet" number. The hand labels alone already
  show the pattern (fewer, larger droplets as condensation goes on), and the
  model makes the same measurement possible for every frame of every
  recording, not just the five that were outlined by hand.

Hyper-parameter search with Optuna and a speed measurement were on the
supervisor's list in 2022 and are included as well. The comparison with the
electrical (impedance) measurement of the comb could not be done: those
logs are not part of the data I have.
