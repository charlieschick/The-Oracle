# Vendored from Edge Impulse's linux-sdk-python (Apache-2.0), trimmed to the
# base ImpulseRunner only - avoids pulling in edge_impulse_linux's package
# init, which hard-imports pyaudio and cv2 (and cv2 is unusable here: the
# board's python-apps-base image pins opencv-python-headless to a custom
# build that isn't published to public PyPI). Image preprocessing is done
# with Pillow instead, in main.py.
import subprocess
import os.path
import tempfile
import shutil
import time
import signal
import socket
import json
from multiprocessing import shared_memory, resource_tracker
import numpy as np


def now():
    return round(time.time() * 1000)


class ImpulseRunner:
    def __init__(self, model_path: str, timeout: int = 30, allow_shm=True):
        self._model_path = model_path
        self._tempdir = None
        self._runner = None
        self._client = None
        self._ix = 0
        self._debug = False
        self._hello_resp = None
        self._allow_shm = allow_shm
        self._input_shm = None
        self._freeform_output_shm = []
        self._timeout = timeout if not allow_shm else None

    def init(self, debug=False):
        if not os.path.exists(self._model_path):
            raise Exception("Model file does not exist: " + self._model_path)

        if not os.access(self._model_path, os.X_OK):
            raise Exception('Model file "' + self._model_path + '" is not executable')

        self._debug = debug
        self._tempdir = tempfile.mkdtemp()
        socket_path = os.path.join(self._tempdir, "runner.sock")
        cmd = [self._model_path, socket_path]
        if debug:
            self._runner = subprocess.Popen(cmd)
        else:
            self._runner = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        while not os.path.exists(socket_path) or self._runner.poll() is not None:
            time.sleep(0.1)

        if self._runner.poll() is not None:
            raise Exception("Failed to start runner (" + str(self._runner.poll()) + ")")

        self._client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._client.settimeout(self._timeout)
        self._client.connect(socket_path)

        hello_resp = self._hello_resp = self.hello()

        if self._allow_shm:
            if 'features_shm' in hello_resp.keys():
                shm_name = hello_resp['features_shm']['name'].lstrip('/')
                shm = shared_memory.SharedMemory(name=shm_name)
                self._input_shm = {
                    'shm': shm,
                    'type': hello_resp['features_shm']['type'],
                    'elements': hello_resp['features_shm']['elements'],
                    'array': np.ndarray((hello_resp['features_shm']['elements'],), dtype=np.float32, buffer=shm.buf)
                }

            if 'freeform_output_shm' in hello_resp.keys():
                for output_shm in hello_resp['freeform_output_shm']:
                    shm_name = output_shm['name'].lstrip('/')
                    shm = shared_memory.SharedMemory(name=shm_name)
                    self._freeform_output_shm.append({
                        'index': output_shm['index'],
                        'shm': shm,
                        'type': output_shm['type'],
                        'elements': output_shm['elements'],
                        'array': np.ndarray((output_shm['elements'],), dtype=np.float32, buffer=shm.buf)
                    })

        return self._hello_resp

    def __del__(self):
        self.stop()

    def stop(self):
        if self._tempdir is not None:
            shutil.rmtree(self._tempdir)
            self._tempdir = None

        if self._client is not None:
            self._client.close()
            self._client = None

        if self._runner is not None:
            os.kill(self._runner.pid, signal.SIGINT)
            self._runner = None

        if self._input_shm is not None:
            self._input_shm['shm'].close()
            resource_tracker.unregister(self._input_shm['shm']._name, "shared_memory")
            self._input_shm = None

        for shm in self._freeform_output_shm:
            shm['shm'].close()
            resource_tracker.unregister(shm['shm']._name, "shared_memory")
        self._freeform_output_shm = []

    def hello(self):
        return self.send_msg({"hello": 1})

    def classify(self, data):
        if self._input_shm:
            self._input_shm['array'][:] = data
            msg = {"classify_shm": {"elements": len(data)}}
        else:
            msg = {"classify": data}

        if self._debug:
            msg["debug"] = True

        send_resp = self.send_msg(msg)

        if 'result' in send_resp and 'freeform' in send_resp['result'] and send_resp['result']['freeform'] == 'shm':
            send_resp['result']['freeform'] = [shm['array'].tolist() for shm in self._freeform_output_shm]

        return send_resp

    def send_msg(self, msg):
        if not self._client:
            raise Exception("ImpulseRunner is not initialized (call init())")

        self._ix += 1
        ix = self._ix
        msg["id"] = ix
        # sendall, not send: classify payloads for a real image (100k+ pixel
        # ints as JSON) are well over a Unix socket's default send buffer, so
        # a plain send() silently under-sends and the runner hangs waiting
        # for the rest of a message that's never coming.
        self._client.sendall(json.dumps(msg).encode("utf-8"))

        data = b""
        while True:
            chunk = self._client.recv(1024)
            if chunk[-1] == 0:
                data = data + chunk[:-1]
                break
            data = data + chunk

        braces_open = 0
        braces_closed = 0
        line = ""
        resp = None
        for c in data.decode("utf-8"):
            if c == "{":
                line += c
                braces_open += 1
            elif c == "}":
                line += c
                braces_closed += 1
                if braces_closed == braces_open:
                    resp = json.loads(line)
            elif braces_open > 0:
                line += c
            if resp is not None:
                break

        if resp is None:
            raise Exception("No data or corrupted data received")
        if resp["id"] != ix:
            raise Exception("Wrong id, expected: " + str(ix) + " but got " + str(resp["id"]))
        if not resp["success"]:
            raise Exception(resp["error"])

        del resp["id"]
        del resp["success"]
        return resp
