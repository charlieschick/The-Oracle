# Tools

Standalone helper tools used while building this project - not part of the
running device.

- **[`detect_dice.py`](detect_dice.py)** - classical computer-vision script that finds dice in a folder of photos and writes out crops + bounding boxes, used to build the training dataset (see [`../training`](../training)). Run it against a folder of photos: `python detect_dice.py path/to/photos`.
- **[`sd_manager.html`](sd_manager.html)** - open directly in a browser to browse, download, and delete files on the Sense's SD card over CircuitPython's Web Workflow, without needing USB.
