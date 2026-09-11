# Sense - Seeed XIAO ESP32S3 Sense (CircuitPython)

Lives in the base of the dice cup. Captures a photo of the two dice on
touch, sends it to the Uno Q (see [`../suziq`](../suziq)), and speaks back
the reading it gets in return.

## Hardware

- Seeed Studio XIAO ESP32S3 Sense (camera + SD card slot built in)
- [Adafruit I2S Amplifier BFF](https://www.adafruit.com/product/5770) (MAX98357, 3W mono), stacked directly onto the XIAO's header pins
- A small 4-ohm oval speaker, wired to the BFF's Picoblade output
- A 10-pixel NeoPixel strip (illuminates the inside of the cup), soldered onto header pins the BFF leaves exposed
- A wire/exposed pad on D3 for touch input, also soldered onto a BFF header pin
- LiPo battery, wired directly to the XIAO's own battery pads (not through the BFF stack), with a 3-pin SPDT switch in-line on the negative lead as a simple on/off

See [`../media`](../media) for the wiring schematic.

## Pin reference

| Pin | Used for |
|---|---|
| D3 | Touch input |
| D7 | NeoPixel data |
| D0 / D1 / D2 | I2S audio out (data / word-select / bit-clock) - used by the amp BFF |
| SPI + SDCS | SD card (on-board) |
| CAM_* | Camera (on-board) |
| BAT+ / BAT- | Battery, with the switch in-line on BAT- |

## Setup

1. Install [CircuitPython](https://circuitpython.org/board/seeed_xiao_esp32s3_sense/) on the board.
2. Copy the required libraries from the [Adafruit CircuitPython Bundle](https://circuitpython.org/libraries) into `/lib` on the board: `adafruit_requests`, `adafruit_connection_manager`.
3. Copy [`settings.toml.example`](settings.toml.example) to `settings.toml` on the board and fill in your WiFi credentials and the Uno Q's address.
4. Copy [`oracle_capture.py`](oracle_capture.py) to `code.py` on the board.

On boot you should hear a short sanity beep (confirms the speaker/amp is
wired correctly, independent of anything else). Touch D3 to trigger a
capture.

## Also in this folder

[`dataset_capture.py`](dataset_capture.py) is the tool used to collect the
training photos for the pip-detection model (see [`../training`](../training))
- touch-to-snap with no networking, saves numbered shots to the SD card.
Not needed for normal operation.
