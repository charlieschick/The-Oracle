# SuziQ - Arduino Uno Q (App Lab)

The "brain": receives a dice photo from the Sense, detects the pips, composes
a reading with a local LLM, and speaks it back - all on-device, no cloud.

## What's here

```
app.yaml              Bricks used: arduino:llm, arduino:web_ui
python/main.py        The whole pipeline - see its module docstring
python/ei_runner.py   Vendored Edge Impulse Linux SDK runner (see below)
python/requirements.txt
sketch/sketch.ino     LED matrix feedback (breathing / busy / result), driven from main.py over the Router Bridge
assets/index.html     Manual test page + reading history browser, served at :7000
models/*.eim          The trained pip-detection model (Edge Impulse, FOMO)
```

## Why a custom model loader instead of the object-detection brick

App Lab's `arduino:object_detection` brick has a model picker meant to pull
custom Edge Impulse models in directly, but it does not reliably register a
model onto the board. `ei_runner.py` is a trimmed copy of Edge Impulse's own
Linux SDK runner class, used to load the `.eim` file directly instead - with
a fix for a bug in that SDK (`socket.send()` silently truncates a large
payload instead of `sendall()`, which makes classification hang on any
real-sized image). See [`../training`](../training) for how the model
itself was built, and the steps to get a `.eim` onto the board this way.

## Setup

1. Install [Arduino App Lab](https://docs.arduino.cc/software/app-lab/) and connect it to your Uno Q.
2. Copy this `suziq/` folder to `~/ArduinoApps/oracle/` on the board (or use App Lab's import).
3. Download a [Piper](https://github.com/OHF-Voice/piper1-gpl) voice (we used `en_GB-alan-medium`, from the [Piper voices on Hugging Face](https://huggingface.co/rhasspy/piper-voices)) and place both the `.onnx` and `.onnx.json` files in `python/voices/`. Not included here - it's a ~60MB stock asset, not project-specific.
4. Train your own pip-detection model (or adapt this one) following [`../training`](../training), and place the resulting `.eim` in `models/`.
5. Start the app in App Lab, or `arduino-app-cli app start ~/ArduinoApps/oracle`.
6. Optional but recommended: `arduino-app-cli properties set default ~/ArduinoApps/oracle` so it auto-starts on boot - otherwise you'll need to start it manually after every reboot.

Once running, `http://<uno-q-address>:7000/` serves a manual test page
(upload a photo, hear the reading, browse history) - useful for confirming
everything works before wiring up the Sense.

## Notes

- The local LLM (`arduino:llm` brick) has a slow first load after every
  restart - `main.py` does a throwaway warmup call at boot so that delay
  never lands on a real touch from the Sense.
- `LargeLanguageModel` enables conversation memory by default. Each reading
  needs to be independent of the last, so every real call is followed by
  `clear_memory()` - without it, the model gradually anchors onto repeating
  an earlier reading regardless of the actual dice rolled.
