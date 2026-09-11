# Oracle capture loop: touch D3 (NeoPixel flash, same as dataset capture)
# snaps a photo, POSTs it to the Uno Q's local /consult endpoint, and plays
# back the spoken reading it returns through the amp.

import time
import array
import gc
import os
import board
import busio
import touchio
import neopixel
import audiobusio
import audiomixer
import audiocore
import espcamera
import sdcardio
import storage
import wifi
import socketpool
import adafruit_requests

# ==================== CONFIG - tune these ====================
CAMERA_FRAME_SIZE = espcamera.FrameSize.SVGA  # 800x600 - matches training data
CAMERA_JPEG_QUALITY = 10
CAMERA_WARMUP_FRAMES = 2  # discarded - ESP32 cams often return torn/partial
                          # frames right after idle

TOUCH_THRESHOLD_MARGIN = 2000

NUM_PIXELS = 10
FLASH_BRIGHTNESS = 0.05  # matches dataset capture, so live shots match training lighting
FLASH_COLOR = (255, 255, 255)

SAMPLE_RATE = 22050       # Piper's output rate
MIXER_LEVEL = 0.7         # 0.0-1.0 output volume via the mixer

SANITY_TONE_FREQ = 440
SANITY_TONE_VOLUME = 0.5
SANITY_TONE_DURATION = 0.5

REQUEST_TIMEOUT = 30  # seconds - detection + LLM + TTS on-device, give it room
# ===============================================================

print("\n=== Oracle v2: camera -> Uno Q /consult -> spoken reading ===")

pixels = neopixel.NeoPixel(board.D7, NUM_PIXELS, brightness=FLASH_BRIGHTNESS, auto_write=True)
pixels.fill((0, 0, 0))

sd = sdcardio.SDCard(board.SPI(), board.SDCS)
vfs = storage.VfsFat(sd)
storage.mount(vfs, "/sd")
print("SD mounted:", os.listdir("/sd"))

cam_i2c = busio.I2C(board.CAM_SCL, board.CAM_SDA)
cam = espcamera.Camera(
    data_pins=board.CAM_DATA,
    pixel_clock_pin=board.CAM_PCLK,
    vsync_pin=board.CAM_VSYNC,
    href_pin=board.CAM_HREF,
    i2c=cam_i2c,
    external_clock_pin=board.CAM_XCLK,
    external_clock_frequency=20_000_000,
    powerdown_pin=None,
    reset_pin=None,
    pixel_format=espcamera.PixelFormat.JPEG,
    frame_size=CAMERA_FRAME_SIZE,
    jpeg_quality=CAMERA_JPEG_QUALITY,
    framebuffer_count=2,
)
print(f"Camera ready: {cam.sensor_name}")

# Capacitive touch baseline drifts with humidity/temperature/what's nearby,
# so it's measured fresh at every boot rather than using a fixed threshold.
touch = touchio.TouchIn(board.D3)
time.sleep(0.5)
touch.threshold = touch.raw_value + TOUCH_THRESHOLD_MARGIN
print(f"Calibrated touch threshold = {touch.threshold} (baseline was {touch.raw_value})")

i2s = audiobusio.I2SOut(bit_clock=board.D2, word_select=board.D1, data=board.D0)
mixer = audiomixer.Mixer(voice_count=1, sample_rate=SAMPLE_RATE, channel_count=1,
                          bits_per_sample=16, samples_signed=False)
i2s.play(mixer)
mixer.voice[0].level = MIXER_LEVEL

# Sanity beep - confirms the speaker/amp/wiring is fine, independent of
# anything network-related, every time this file (re)runs.
import math
_tone_len = SAMPLE_RATE // SANITY_TONE_FREQ
_tone = array.array("H", [0] * _tone_len)
for _i in range(_tone_len):
    _tone[_i] = int((math.sin(math.pi * 2 * _i / _tone_len) * SANITY_TONE_VOLUME + 1) * (2 ** 15))
mixer.voice[0].play(audiocore.RawSample(_tone, sample_rate=SAMPLE_RATE), loop=True)
time.sleep(SANITY_TONE_DURATION)
mixer.voice[0].stop()
print("(That beep was the speaker sanity check.)")

if not wifi.radio.ipv4_address:
    print("WiFi not yet connected, waiting...")
    for _ in range(50):
        if wifi.radio.ipv4_address:
            break
        time.sleep(0.3)
if wifi.radio.ipv4_address:
    print(f"WiFi connected, IP={wifi.radio.ipv4_address}")
else:
    print("WiFi FAILED to connect - Oracle requests will fail")

pool = socketpool.SocketPool(wifi.radio)
requests = adafruit_requests.Session(pool)

# Set in settings.toml (see settings.toml.example), not hardcoded - point
# this at your Uno Q's /consult endpoint, e.g. "http://<uno-q-ip>:7000/consult".
ORACLE_URL = os.getenv("ORACLE_URL")


# CircuitPython has no urllib, so the reading text that comes back
# URL-encoded in a response header needs decoding by hand.
def url_decode(s):
    out = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c == "%" and i + 2 < len(s):
            out += chr(int(s[i + 1:i + 3], 16))
            i += 3
        elif c == "+":
            out += " "
            i += 1
        else:
            out += c
            i += 1
    return out


def ask_oracle(jpeg_data):
    headers = {"Content-Type": "image/jpeg"}
    response = requests.post(ORACLE_URL, data=bytes(jpeg_data), headers=headers, timeout=REQUEST_TIMEOUT)
    try:
        if response.status_code != 200:
            print(f"  Oracle error {response.status_code}: {response.text}")
            return None
        reality = response.headers.get("x-reality", "")
        spiritual = response.headers.get("x-spiritual", "")
        reading = url_decode(response.headers.get("x-reading", ""))
        print(f"  Reality: {reality}  Spiritual: {spiritual}")
        print(f"  Reading:\n{reading}")
        return {"audio": response.content, "reality": reality, "spiritual": spiritual, "reading": reading}
    finally:
        response.close()


def play_bell():
    # A short decaying "ding" - played right before the reading, so the
    # speaking doesn't start abruptly out of nowhere after the network lag.
    freq = 880
    duration = 0.8
    n = int(SAMPLE_RATE * duration)
    tone = array.array("H", [0] * n)
    for i in range(n):
        t = i / SAMPLE_RATE
        envelope = math.exp(-3.5 * t)
        tone[i] = int((math.sin(math.pi * 2 * freq * t) * envelope * 0.6 + 1) * (2 ** 15))
    mixer.voice[0].play(audiocore.RawSample(tone, sample_rate=SAMPLE_RATE))
    while mixer.voice[0].playing:
        time.sleep(0.02)


def play_wav_bytes(wav_data):
    # Skip the 44-byte WAV header; reinterpret as signed 16-bit, then shift
    # to our unsigned/32768-centered convention for the Mixer.
    pcm_bytes = bytes(wav_data[44:])
    signed_samples = array.array("h", pcm_bytes)
    samples = array.array("H", [0] * len(signed_samples))
    for i in range(len(signed_samples)):
        samples[i] = signed_samples[i] + 32768
    sample = audiocore.RawSample(samples, sample_rate=SAMPLE_RATE)
    mixer.voice[0].play(sample)
    while mixer.voice[0].playing:
        time.sleep(0.05)


print(f"Free memory at start: {gc.mem_free()} bytes")
print("Ready. Touch D3 to consult the Oracle.")

while True:
    if touch.value:
        pixels.fill(FLASH_COLOR)
        for _ in range(CAMERA_WARMUP_FRAMES):
            cam.take()
        jpeg = cam.take()
        pixels.fill((0, 0, 0))

        if jpeg is None:
            print("  capture FAILED, try again")
        else:
            with open("/sd/capture.jpg", "wb") as f:
                f.write(jpeg)
            print(f"  saved /sd/capture.jpg ({len(jpeg)} bytes)")

            print("Consulting the Oracle...")
            t0 = time.monotonic()
            try:
                result = ask_oracle(jpeg)
            except Exception as e:
                result = None
                print(f"  Oracle request FAILED: {e}")
            print(f"  round trip: {time.monotonic() - t0:.2f}s")

            if result:
                try:
                    with open("/sd/reading.wav", "wb") as f:
                        f.write(result["audio"])
                except Exception as e:
                    print(f"  reading save FAILED: {e}")
                play_bell()
                play_wav_bytes(result["audio"])
                print("  playback: done")

        while touch.value:  # wait for release, simple debounce
            time.sleep(0.02)
        time.sleep(0.3)

        gc.collect()
        print("Ready. Touch D3 to consult the Oracle.")
    time.sleep(0.02)
