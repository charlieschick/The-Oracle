# Training the pip-detection model

The dice-reading model was built in [Edge Impulse](https://edgeimpulse.com)
Studio. This is a walkthrough of the steps, not a script - if you want API
access to automate parts of this, Edge Impulse's own REST API and Python
SDK (`edgeimpulse_api`) are well documented and you'll be able to use them
directly; that's beyond what we're covering here.

## 1. Capture photos

Use [`../sense/dataset_capture.py`](../sense/dataset_capture.py) on the
Sense to collect photos of real dice rolls, from the same camera position
and lighting the finished device will actually use. We used around 300
photos total, gathered in a few batches, making sure all six pip values
were roughly equally represented and dice appeared at varied positions and
angles - a lopsided dataset (e.g. mostly showing 1-3) will train a model
that's noticeably worse at reading 4-6.

## 2. Generate bounding boxes

[`../tools/detect_dice.py`](../tools/detect_dice.py) is a classical
computer-vision tool (OpenCV, no ML) that finds the dice in each photo and
writes out a crop + bounding box per die. It's tuned for this specific
setup (dice on a red-felt-lined enclosure) - the `MIN_SOLIDITY`,
`RED_HUE_LOW/HIGH`, and area-fraction constants at the top will need
retuning for a different background or camera.

It won't get every photo right - review its `debug/` output (boxes drawn
on the original images) and fix stragglers by hand before uploading.

## 3. Label the pips

Classical CV can find *where* the dice are, but not reliably *what* they
show - that part is manual. Open each crop and record its pip count. This
is genuinely the slow part; budget real time for it, especially since
you'll likely repeat this over a few rounds as you add more data.

One lesson worth passing on: reading pips from a small multi-image contact
sheet (several crops at once) is measurably less reliable than reviewing
one full-size crop at a time - worth a spot-check of your own labels before
trusting them, however you end up doing this step.

## 4. Upload to Edge Impulse and label

Create a project in Edge Impulse Studio, set it up as an **Object
Detection** project, and upload your images with their bounding boxes and
labels (drag-and-drop through Studio's own UI is the straightforward path;
Edge Impulse's REST API also supports this directly, if you'd rather script
it). Edge Impulse now also offers an AI-assisted labeling tool in Studio,
which we didn't end up trying.

## 5. Build the impulse

- **Image** processing block, resized to what your target model expects (we used 320x320, squash resize mode).
- **Object Detection (Images)** learning block, using the **FOMO (Faster Objects, More Objects)** architecture - a lightweight, centroid-based detector well suited to a small board with no NPU.
- Train. Studio reports precision/recall/F1 per class after each run; we landed on F1 ≈ 0.92 after a few rounds of adding data and fixing bad labels. If accuracy plateaus below what you need, check your label quality before adding more images - a clean, smaller dataset outperformed a larger, noisier one for us by a wide margin.

## 6. Export for the Uno Q

In Studio's Deployment tab, build for **Linux (AARCH64)** (`runner-linux-aarch64`)
- that's the correct target for a plain Uno Q (no NPU). Download the
resulting `.eim` file and place it in
[`../suziq/models/`](../suziq/models/).

## A gotcha worth knowing about

FOMO doesn't apply non-max suppression to its output, so two dice sitting
close together (or even one die alone, sometimes) can produce two
overlapping/duplicate detections a few pixels apart instead of one. Left
unhandled, this shows up as "can't find exactly two dice" on photos that
are perfectly readable. See `_dedupe_boxes()` in
[`../suziq/python/main.py`](../suziq/python/main.py) for the fix: collapse
detections whose centers are within a small pixel distance of a
higher-confidence one before counting boxes.
