# Dice dataset capture: used to build the training set for the pip-detection
# model. Touch D3, release to snap a photo (with NeoPixel flash), saved as
# /sd/dataset/dice_NNNN.jpg (auto-incrementing, resumes numbering across
# sessions so repeated runs don't clobber earlier shots). Stripped down from
# the real Oracle loop (oracle_capture.py), just camera, so you can roll and 
# snap quickly.
#
# Same camera settings as production (SVGA, quality 10, 2 warmup frames
# discarded - ESP32 cams commonly return torn/partial frames right after
# idle) so this data matches what the deployed camera will actually produce.
#
# This is a one-time data-collection tool, not something that runs alongside
# the production loop - copy oracle_capture.py to code.py on the board to go
# back to normal operation once you've captured enough shots.
import time
import os
import board
import busio
import touchio
import neopixel
import espcamera
import sdcardio
import storage

CAMERA_FRAME_SIZE = espcamera.FrameSize.SVGA  # 800x600
CAMERA_JPEG_QUALITY = 10
CAMERA_WARMUP_FRAMES = 2

TOUCH_THRESHOLD_MARGIN = 2000

NUM_PIXELS = 10
FLASH_BRIGHTNESS = 0.05  # lowest value that looked good in testing - retune if needed
FLASH_COLOR = (255, 255, 255)

print("\n=== Dice dataset capture (NeoPixel flash) ===")

pixels = neopixel.NeoPixel(board.D7, NUM_PIXELS, brightness=FLASH_BRIGHTNESS, auto_write=True)
pixels.fill((0, 0, 0))

sd = sdcardio.SDCard(board.SPI(), board.SDCS)
vfs = storage.VfsFat(sd)
storage.mount(vfs, "/sd")

if "dataset" not in os.listdir("/sd"):
    os.mkdir("/sd/dataset")

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

touch = touchio.TouchIn(board.D3)
time.sleep(0.5)
touch.threshold = touch.raw_value + TOUCH_THRESHOLD_MARGIN
print(f"Calibrated touch threshold = {touch.threshold} (baseline was {touch.raw_value})")

# Resume numbering where a previous session left off.
existing = [f for f in os.listdir("/sd/dataset") if f.startswith("dice_") and f.endswith(".jpg")]
shot_num = 0
for f in existing:
    try:
        shot_num = max(shot_num, int(f[5:9]))
    except ValueError:
        pass

print(f"{len(existing)} existing shots found. Resuming at #{shot_num + 1}.")
print("Roll the dice, touch D3, release to capture. Repeat.")

while True:
    if touch.value:
        pixels.fill(FLASH_COLOR)
        for _ in range(CAMERA_WARMUP_FRAMES):
            cam.take()
        jpeg = cam.take()
        pixels.fill((0, 0, 0))
        if jpeg is not None:
            shot_num += 1
            path = f"/sd/dataset/dice_{shot_num:04d}.jpg"
            with open(path, "wb") as f:
                f.write(jpeg)
            print(f"  saved {path} ({len(jpeg)} bytes)")
        else:
            print("  capture FAILED, try again")
        while touch.value:  # wait for release, simple debounce
            time.sleep(0.02)
        time.sleep(0.3)
    time.sleep(0.02)
