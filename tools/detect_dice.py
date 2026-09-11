"""Find dice bounding boxes in a folder of photos using classical CV.

Deliberately not a neural net: dice are small, high-contrast, roughly-square
objects against a fairly uniform background, which plain thresholding +
contour detection handles well without any training data of its own.

For each input image, writes:
  - crops/<name>_<i>.jpg       cropped region for each detected die (for the
                                pip-counting step, done separately)
  - debug/<name>.jpg           original image with detected boxes drawn on,
                                so box quality can be sanity-checked visually
  - boxes.json                 {image_name: [{x, y, width, height}, ...]}
                                in Edge Impulse's pixel-coordinate convention
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Tuning knobs - dice are assumed to occupy a moderate fraction of the frame,
# not full-image and not tiny speckle noise.
MIN_AREA_FRAC = 0.001   # a die smaller than this fraction of the image area is noise
MAX_AREA_FRAC = 0.25    # bigger than this is probably background, not a die
MIN_SOLIDITY = 0.80     # dice are convex-ish; filters out stringy/irregular blobs
# (0.85 rejected too many legit dice in batch 2, apparently blurrier/softer
# silhouettes than batch 1 - swept 0.55-0.85 against both batches; 0.80 is the
# lowest value with zero false positives on batch 1, below that batch 1 starts
# picking up spurious "3+" detections from rim/shadow noise)
PAD_FRAC = 0.08         # pad each crop a bit so we don't clip the die's edge

# The enclosure's out-of-focus rim is a strongly red/saturated ring around the
# in-focus wood disc. It's not a fixed size or position shot-to-shot (foam
# shifts when the compartment closes), so rather than masking a fixed circle,
# mask by color: pixels near red hue with real saturation get excluded before
# thresholding, so the rim never merges into a contour or skews Otsu's split.
RED_HUE_LOW = 12         # OpenCV hue is 0-179; red wraps around 0/180
RED_HUE_HIGH = 160
RED_SAT_MIN = 90
BORDER_DILATE = 9        # px; eat a bit into the blurry rim/wood gradient too

EDGE_MARGIN = 3           # px; how close counts as "touching" the frame edge

# Red/blue labeling: left-right is the primary axis since the two dice are
# rarely stacked exactly vertically; only fall back to top-bottom when the
# x-centers are close enough that "leftmost" would be a coin flip.
TIE_FRAC = 0.15         # x-centers within this fraction of image width count as tied
RED = (0, 0, 255)       # BGR
BLUE = (255, 0, 0)      # BGR


def order_red_blue(boxes, img_width):
    """Return boxes sorted so index 0 is "red" (left, or top if x is tied)."""
    if len(boxes) < 2:
        return boxes
    centers = [(x + bw / 2, y + bh / 2) for x, y, bw, bh in boxes]
    x_centers = [c[0] for c in centers]
    if max(x_centers) - min(x_centers) < TIE_FRAC * img_width:
        key = lambda i: centers[i][1]  # tied on x -> sort top to bottom
    else:
        key = lambda i: centers[i][0]  # sort left to right
    order = sorted(range(len(boxes)), key=key)
    return [boxes[i] for i in order]


def find_dice_boxes(img):
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, _ = cv2.split(hsv)
    red_mask = (((hue < RED_HUE_LOW) | (hue > RED_HUE_HIGH)) & (sat > RED_SAT_MIN))
    red_mask = red_mask.astype(np.uint8) * 255
    red_mask = cv2.dilate(red_mask, np.ones((BORDER_DILATE, BORDER_DILATE), np.uint8))
    not_border = cv2.bitwise_not(red_mask)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu picks its own threshold - robust to the exact lighting/exposure
    # of a given shot rather than needing a hand-tuned brightness cutoff.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Try both polarities - depending on whether the dice are lighter or
    # darker than the background, the "object" could be the 0s or the 1s.
    candidates = []
    for variant in (thresh, cv2.bitwise_not(thresh)):
        variant = cv2.bitwise_and(variant, not_border)  # drop the red rim before contouring
        closed = cv2.morphologyEx(variant, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(contours)

    img_area = h * w
    boxes = []
    for c in candidates:
        area = cv2.contourArea(c)
        if not (MIN_AREA_FRAC * img_area <= area <= MAX_AREA_FRAC * img_area):
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / bh if bh else 0
        if not (0.6 <= aspect <= 1.6):  # dice are roughly square, even tilted
            continue

        # Reject blobs hugging a corner (two adjacent edges) - that's the
        # enclosure rim/glare, not a die. A die can legitimately be cut off by
        # ONE edge (the framing shifts shot to shot as the foam settles), so
        # only two-edges-at-once is treated as an artifact.
        touches = {
            "l": x <= EDGE_MARGIN, "r": x + bw >= w - EDGE_MARGIN,
            "t": y <= EDGE_MARGIN, "b": y + bh >= h - EDGE_MARGIN,
        }
        n_touch = sum(touches.values())
        if n_touch >= 2 and not (touches["t"] and touches["b"]) and not (touches["l"] and touches["r"]):
            continue

        boxes.append((x, y, bw, bh))

    # De-duplicate near-identical boxes from the two threshold polarities.
    deduped = []
    for b in boxes:
        bx, by, bw, bh = b
        bcx, bcy = bx + bw / 2, by + bh / 2
        if any(abs(bcx - (ox + ow / 2)) < ow * 0.5 and abs(bcy - (oy + oh / 2)) < oh * 0.5
               for ox, oy, ow, oh in deduped):
            continue
        deduped.append(b)
    return deduped


def main(src_dir: str):
    src = Path(src_dir)
    crops_dir = src / "crops"
    debug_dir = src / "debug"
    crops_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    all_boxes = {}
    image_paths = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  could not read {path.name}, skipping")
            continue
        boxes = find_dice_boxes(img)
        print(f"{path.name}: found {len(boxes)} candidate box(es)")

        debug_img = img.copy()
        h, w = img.shape[:2]
        boxes = order_red_blue(boxes, w)
        entry = []
        for i, (x, y, bw, bh) in enumerate(boxes):
            color_name = "red" if i == 0 else "blue" if i == 1 else "extra"
            draw_color = RED if color_name == "red" else BLUE if color_name == "blue" else (0, 255, 0)

            pad_x, pad_y = int(bw * PAD_FRAC), int(bh * PAD_FRAC)
            cx0, cy0 = max(0, x - pad_x), max(0, y - pad_y)
            cx1, cy1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
            crop = img[cy0:cy1, cx0:cx1]
            crop_name = f"{path.stem}_{color_name}.jpg"
            cv2.imwrite(str(crops_dir / crop_name), crop)

            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), draw_color, 2)
            cv2.putText(debug_img, color_name, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)

            entry.append({"crop": crop_name, "color": color_name, "x": x, "y": y, "width": bw, "height": bh})

        cv2.imwrite(str(debug_dir / path.name), debug_img)
        all_boxes[path.name] = entry

    with open(src / "boxes.json", "w") as f:
        json.dump(all_boxes, f, indent=2)
    print(f"\nWrote {src / 'boxes.json'}, {len(image_paths)} image(s) processed.")
    print(f"Check {debug_dir}/ to sanity-check box placement before trusting this.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
