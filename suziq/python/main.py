"""Oracle - Arduino App Lab app, running on the Uno Q.

Exposes one main HTTP endpoint, POST /consult: takes a JPEG photo of the two
dice, runs the locally-trained Edge Impulse object-detection model to read
their pip values, composes a short reflective reading with a local LLM
(arduino:llm brick), synthesizes it to speech with Piper, and returns the
WAV. The Sense (sense/oracle_capture.py) is the client that calls this.

Also exposes GET /history (+ /history/{id}/image, /history/{id}/audio) for
a small debugging web page (see ../assets/index.html) that lists past
readings - successes and misses alike - with their photos and audio.

The object-detection model is loaded directly via ei_runner.py (a trimmed,
locally-fixed copy of Edge Impulse's own Linux SDK runner) rather than the
arduino:object_detection brick, because that brick's custom-model picker
does not reliably register an Edge Impulse model onto the board. See
../training for the manual .eim deployment steps this app expects instead.

LED matrix feedback (breathing / busy / dice values / poem scroll) is
driven from here over the Router Bridge - see ../sketch/sketch.ino for the
MCU side.
"""
from datetime import datetime, UTC
from pathlib import Path
from urllib.parse import quote
import io
import threading
import time
import wave

import numpy as np
from PIL import Image
from fastapi import Request
from fastapi.responses import Response

from ei_runner import ImpulseRunner
from arduino.app_utils import App, Bridge, Frame
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_bricks.dbstorage_sqlstore import SQLStore
from piper import PiperVoice

# 5x7 bitmap font for the LED matrix: digits, A-Z, and basic punctuation -
# enough to show the dice values and scroll the poem itself.
FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "01010", "00100", "01010", "10001", "00000", "00000"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    " ": ["00000"] * 7,
    ",": ["00000", "00000", "00000", "00000", "00110", "00100", "01000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "'": ["00100", "00100", "01000", "00000", "00000", "00000", "00000"],
}

MATRIX_COLS = 13
CHAR_W = 6  # 5-wide glyph + 1 column of spacing
SCROLL_STEP_S = 0.06


def _stamp(canvas: np.ndarray, ch: str, col_offset: int, brightness: int = 7):
    glyph = FONT.get(ch.upper(), FONT[" "])
    for r, row in enumerate(glyph):
        for c, bit in enumerate(row):
            if bit == "1":
                canvas[r, col_offset + c] = brightness


def render_static_pair(left: str, right: str) -> bytes:
    """13-wide static frame: two characters side by side, no scrolling -
    used for the dice values, which hang on screen through the LLM/TTS
    'thinking' phase rather than scrolling."""
    canvas = np.zeros((8, MATRIX_COLS), dtype=np.uint8)
    _stamp(canvas, left, 1)
    _stamp(canvas, right, 7)
    return Frame(canvas).to_board_bytes()


def _scroll_frames(text: str) -> list[bytes]:
    pad = MATRIX_COLS
    width = pad + len(text) * CHAR_W + pad
    canvas = np.zeros((8, width), dtype=np.uint8)
    for i, ch in enumerate(text):
        _stamp(canvas, ch, pad + i * CHAR_W)
    return [
        Frame(canvas[:, offset:offset + MATRIX_COLS]).to_board_bytes()
        for offset in range(width - MATRIX_COLS + 1)
    ]


# Generation counter: each new consult bumps this, and any in-flight scroll
# checks it every frame so a fresh touch cancels an old poem mid-scroll
# instead of the two fighting over the display.
_generation = 0
_generation_lock = threading.Lock()


def next_generation() -> int:
    global _generation
    with _generation_lock:
        _generation += 1
        return _generation


def _current_generation() -> int:
    with _generation_lock:
        return _generation


def show_dice_values(left: str, right: str):
    Bridge.notify("show_result", render_static_pair(left, right))


MISS_HOLD_S = 2.5


def show_miss_then_revert(gen: int):
    """Show X X for a beat (there's no poem to scroll for a miss, so
    something has to give it dwell time), then back to breathing - unless a
    newer consult has already taken over the display."""
    def run():
        time.sleep(MISS_HOLD_S)
        if _current_generation() == gen:
            Bridge.notify("resume_breathing")

    threading.Thread(target=run, daemon=True).start()


def play_poem_scroll(text: str, gen: int, passes: int = 2):
    """Scroll the poem text across the matrix, twice, in the background.
    Bails out early - without touching resume_breathing - if a newer
    consult has started, so the next request's own display takes over
    cleanly instead of this one clobbering it on its way out."""
    frames = _scroll_frames(text.replace("\n", "   "))

    def run():
        for _pass in range(passes):
            for frame in frames:
                if _current_generation() != gen:
                    return
                Bridge.notify("show_result", frame)
                time.sleep(SCROLL_STEP_S)
        if _current_generation() == gen:
            Bridge.notify("resume_breathing")

    threading.Thread(target=run, daemon=True).start()


VOICE_PATH = Path(__file__).parent / "voices" / "en_GB-alan-medium.onnx"
MODEL_PATH = Path(__file__).parent.parent / "models" / "oracle-linux-aarch64-v1-dice-pip-detector.eim"
PAUSE_MS = 900
CONFIDENCE_THRESHOLD = 0.5
TIE_FRAC = 0.15  # x-centers within this fraction of image width count as tied -> sort top-to-bottom

REALITY_MEANINGS = ["Me", "Other", "Loved One", "Nature", "Work", "Quiet"]
SPIRITUAL_MEANINGS = ["Inside", "Outside", "Sublime", "Past", "Future", "Now"]

SYSTEM_PROMPT = """You are the Oracle: a reflection tool built on two thrown dice, not a
fortune-teller. Compose a three-line, haiku-like reflection - concrete, sensory images (a
texture, a sound, a small action), not abstract feelings. Show, do not state.

Reality die (1-6): Me, Other, Loved One, Nature, Work, Quiet.
Spiritual die (1-6): Inside, Outside, Sublime, Past, Future, Now.

Line 1: image from the Reality die meaning. Line 2: image from the Spiritual die meaning.
Line 3: open and ambiguous.

Respond with exactly three lines of the reflection itself. Never repeat the dice values,
and add no title, preamble, or explanation."""


def init_llm():
    return LargeLanguageModel(system_prompt=SYSTEM_PROMPT, temperature=0.8)


def init_voice():
    return PiperVoice.load(str(VOICE_PATH))


def init_dice_detector():
    runner = ImpulseRunner(str(MODEL_PATH), allow_shm=False)
    info = runner.init()
    return runner, info["model_parameters"]


def init_history_db():
    db = SQLStore("oracle_history.db")
    db.start()
    db.create_or_replace_table("readings", {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "timestamp": "TEXT",
        "status": "TEXT",       # "ok" or "no_detection"
        "reality": "INTEGER",
        "spiritual": "INTEGER",
        "num_boxes": "INTEGER",  # confident boxes found - diagnostic for misses
        "reading": "TEXT",
        "image": "BLOB",
        "audio": "BLOB",
    }, force_drop_table=True)
    return db


llm = init_llm()
llm.chat("hi")  # forces the model to load now, at boot, instead of on the first real touch
llm.clear_memory()  # CloudLLM enables conversation memory (last 10 msgs) by default -
                     # each reading must be independent, not colored by the warmup or
                     # prior rolls, so every chat() call below is followed by a clear
voice = init_voice()
detector, detector_params = init_dice_detector()
history_db = init_history_db()


def synth_reading_wav(text: str, pause_ms: int = PAUSE_MS) -> bytes:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    buf = io.BytesIO()
    wav_file = wave.open(buf, "wb")
    header_set = False
    sample_rate = 22050
    for i, line in enumerate(lines):
        for chunk in voice.synthesize(line):
            if not header_set:
                wav_file.setnchannels(chunk.sample_channels)
                wav_file.setsampwidth(chunk.sample_width)
                wav_file.setframerate(chunk.sample_rate)
                header_set = True
                sample_rate = chunk.sample_rate
            wav_file.writeframes(chunk.audio_int16_bytes)
        if i < len(lines) - 1:
            silence_frames = int(sample_rate * pause_ms / 1000)
            wav_file.writeframes(b"\x00\x00" * silence_frames)
    wav_file.close()
    return buf.getvalue()


def image_to_features(image_bytes: bytes) -> list[int]:
    """Resize/crop a photo to the model's expected input and pack it into the
    flat (R<<16)+(G<<8)+B integer-per-pixel list Edge Impulse's runner expects.
    Mirrors the resize mode the model was trained with (studio default is
    'squash' if not otherwise reported).
    """
    width = detector_params["image_input_width"]
    height = detector_params["image_input_height"]
    mode = detector_params.get("image_resize_mode", "not-reported")
    if mode == "not-reported":
        mode = "squash"

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    in_w, in_h = img.size

    if mode == "fit-shortest":
        aspect = width / height
        if in_w / in_h > aspect:
            new_w = int(in_h * aspect)
            offset = (in_w - new_w) // 2
            img = img.crop((offset, 0, offset + new_w, in_h))
        else:
            new_h = int(in_w / aspect)
            offset = (in_h - new_h) // 2
            img = img.crop((0, offset, in_w, offset + new_h))
        img = img.resize((width, height), Image.BILINEAR)
    elif mode == "fit-longest":
        scale = min(width / in_w, height / in_h)
        new_w, new_h = int(in_w * scale), int(in_h * scale)
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        img = Image.new("RGB", (width, height))
        img.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    else:  # squash
        img = img.resize((width, height), Image.BILINEAR)

    pixels = list(img.getdata())
    return [(r << 16) + (g << 8) + b for r, g, b in pixels]


DEDUPE_DIST = 30  # pixels in the model's 320x320 input space. FOMO sometimes
                   # fires on two adjacent grid cells for one die, producing a
                   # near-duplicate box a few pixels away - collapse those
                   # (keeping the higher-confidence one) before counting boxes.
                   # Genuinely separate dice, even touching ones, sit farther
                   # apart than this in practice.


def _dedupe_boxes(boxes):
    boxes = sorted(boxes, key=lambda b: b["value"], reverse=True)
    kept = []
    for b in boxes:
        bx, by = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
        if all(
            ((bx - (k["x"] + k["width"] / 2)) ** 2 + (by - (k["y"] + k["height"] / 2)) ** 2) ** 0.5 >= DEDUPE_DIST
            for k in kept
        ):
            kept.append(b)
    return kept


def detect_dice(image_bytes: bytes):
    """Read two dice pip values from a photo.

    Returns (values, boxes) where values is [reality, spiritual] (left-to-right,
    or top-to-bottom if the dice are side by side) when exactly two confident
    detections are found, else (None, boxes) with whatever boxes did pass
    threshold so the caller can explain what went wrong.
    """
    features = image_to_features(image_bytes)
    res = detector.classify(features)
    boxes = [b for b in res["result"]["bounding_boxes"] if b["value"] >= CONFIDENCE_THRESHOLD]
    boxes = _dedupe_boxes(boxes)

    if len(boxes) != 2:
        return None, boxes

    width = detector_params["image_input_width"]
    centers = [(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2) for b in boxes]
    x_centers = [c[0] for c in centers]
    if max(x_centers) - min(x_centers) < TIE_FRAC * width:
        order = sorted(range(2), key=lambda i: centers[i][1])  # tied on x -> top to bottom
    else:
        order = sorted(range(2), key=lambda i: centers[i][0])  # left to right

    values = [int(boxes[order[0]]["label"]), int(boxes[order[1]]["label"])]
    return values, boxes


async def consult_handler(request: Request) -> Response:
    """POST a JPEG photo of the two dice, get back a spoken reading as WAV.

    On success: 200, audio/wav body, with X-Reality/X-Spiritual/X-Reading
    headers (URL-encoded - see the url_decode() helper in
    sense/oracle_capture.py, which reads these on the other end).
    On a bad/ambiguous photo: 422, plain text explaining what was seen.
    """
    gen = next_generation()
    Bridge.notify("show_busy")

    image_bytes = await request.body()
    values, boxes = detect_dice(image_bytes)
    if values is None:
        show_dice_values("X", "X")
        history_db.store("readings", {
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "no_detection",
            "num_boxes": len(boxes),
            "image": image_bytes,
        }, create_table=False)
        show_miss_then_revert(gen)
        return Response(
            content=f"Didn't find exactly two confident dice (found {len(boxes)} above "
                     f"{CONFIDENCE_THRESHOLD:.0%} confidence).",
            status_code=422,
            media_type="text/plain",
        )

    reality, spiritual = values
    # Numbers hang on the matrix through the LLM/TTS "thinking" time below,
    # rather than a quick flash - doubles as a working indicator.
    show_dice_values(str(reality), str(spiritual))
    reading = llm.chat(f"Reality die shows {reality}. Spiritual die shows {spiritual}.").strip()
    llm.clear_memory()
    wav_bytes = synth_reading_wav(reading)
    play_poem_scroll(reading, gen)

    history_db.store("readings", {
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "ok",
        "reality": reality,
        "spiritual": spiritual,
        "num_boxes": len(boxes),
        "reading": reading,
        "image": image_bytes,
        "audio": wav_bytes,
    }, create_table=False)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Reality": str(reality),
            "X-Spiritual": str(spiritual),
            "X-Reading": quote(reading),
        },
    )


def history_handler():
    rows = history_db.read(
        "readings",
        columns=["id", "timestamp", "status", "reality", "spiritual", "num_boxes", "reading"],
        order_by="id DESC",
        limit=50,
    )
    return {"readings": rows}


async def history_audio_handler(id: int) -> Response:
    rows = history_db.read("readings", columns=["audio"], condition=f"id = {id}", limit=1)
    if not rows or rows[0]["audio"] is None:
        return Response(status_code=404)
    return Response(content=rows[0]["audio"], media_type="audio/wav")


async def history_image_handler(id: int) -> Response:
    rows = history_db.read("readings", columns=["image"], condition=f"id = {id}", limit=1)
    if not rows or rows[0]["image"] is None:
        return Response(status_code=404)
    return Response(content=rows[0]["image"], media_type="image/jpeg")


web_ui = WebUI()
web_ui.expose_api("POST", "/consult", consult_handler)
web_ui.expose_api("GET", "/history", history_handler)
web_ui.expose_api("GET", "/history/{id}/image", history_image_handler)
web_ui.expose_api("GET", "/history/{id}/audio", history_audio_handler)
web_ui.start()

App.run()
