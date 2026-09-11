# Oracle

![Oracle hero image](media/hero%20image.png)

A reflective physical-computing dice cup. Roll two dice inside it, touch a
sensor, and a locally-run vision model reads the pips, a locally-run LLM
composes a short evocative reading, and it's spoken back through a speaker
in the cup - all on-device, fully offline, no cloud services involved.

Built for the [Arduino UNO Q](https://docs.arduino.cc/hardware/uno-q/)
"Invent the Future with Arduino UNO Q and App Lab" contest.

**Full project writeup, photos, and video:** [The Oracle on Hackster.io](https://www.hackster.io/molecularist/the-oracle-a-kinetically-augmented-device-for-reflection-b593b6)

## How it works

```
 dice cup (Sense)                         Uno Q (SuziQ)
 ─────────────────                        ─────────────────
 touch → camera capture                   /consult endpoint
     │                                         │
     └── POST photo ────────────────────────▶  │
                                                ▼
                                    Edge Impulse model (FOMO)
                                    reads pip values from the photo
                                                │
                                                ▼
                                    local LLM (Gemma, via arduino:llm)
                                    composes a 3-line reading
                                                │
                                                ▼
                                    Piper TTS synthesizes speech
                                                │
     ◀── WAV audio ─────────────────────────────┘
     │
 plays through the speaker
```

Along the way, SuziQ's onboard LED matrix shows live feedback (breathing
while idle, a flash while a photo's being processed, the dice values while
composing, the poem scrolling once it's ready), and a small web page
(`http://<uno-q>:7000/`) lets you browse the history of past readings -
including the ones it couldn't read - with their photos and audio.

## Repo layout

| Folder | What's there |
|---|---|
| [`sense/`](sense) | CircuitPython code for the Seeed XIAO ESP32S3 Sense - camera, touch, speaker, the physical cup electronics |
| [`suziq/`](suziq) | The Arduino App Lab app running on the Uno Q - detection, LLM, TTS, LED matrix, web UI |
| [`tools/`](tools) | Standalone tools used while building this: the OpenCV auto-labeling script, an SD card browser |
| [`training/`](training) | Walkthrough of how the dice-reading model was trained in Edge Impulse |
| [`media/`](media) | Wiring schematic |

Each folder has its own README with setup details.

## Bill of materials

**Hardware**
- Seeed Studio XIAO ESP32S3 Sense (camera + SD card built in)
- Arduino UNO Q
- Adafruit I2S Amplifier BFF (MAX98357)
- Small 4-ohm oval speaker
- 10-pixel NeoPixel strip
- LiPo battery + SPDT switch
- A dice cup with a hollow base, two dice

**Software**
- [CircuitPython](https://circuitpython.org/) (Sense)
- [Arduino App Lab](https://docs.arduino.cc/software/app-lab/) (Uno Q)
- [Edge Impulse](https://edgeimpulse.com/) - FOMO object-detection model for reading dice pips
- [Piper](https://github.com/OHF-Voice/piper1-gpl) - local text-to-speech
- Gemma 3 1B, running locally via App Lab's `arduino:llm` brick (llama.cpp)
- FastAPI (via App Lab's `arduino:web_ui` brick), SQLite (via `arduino:dbstorage_sqlstore`)
- OpenCV - offline, used only to help build the training dataset, never at runtime
- [Claude Code](https://claude.com/claude-code) - used throughout for development, debugging, and deployment

## License

[MIT](LICENSE)

## Acknowledgments

Built with [Claude Code](https://claude.com/claude-code) as a genuine
collaborator throughout - see the project writeup for more on how that
went.
