"""Live backend auth smoke: POST /api/events requires HMAC, ack requires JWT.

Spawns dashboard/backend/routes/events.js on port 3998 with a temp node
process. Cleans up its own frame_ids afterwards. Skips if node deps missing.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import urllib.error

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard", "backend")
DB_PATH = os.path.normpath(os.path.join(BACKEND_DIR, "db", "infraedge.db"))
PORT = 3998

SERVER_JS = r"""
const express = require('express');
const app = express();
app.use(express.json());
app.use('/', require('./routes/events'));
app.listen(3998, () => console.log('smoke-ready'));
"""

SMOKE_IDS = ["smoke-hmac-ok-001"]


def _sig(payload):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
    from mqtt_client import generate_payload_signature
    return generate_payload_signature(payload)


def _post(path, body, token=None):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://localhost:{PORT}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


@unittest.skipIf(shutil.which("node") is None, "node not available")
class TestEventsAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(os.path.join(BACKEND_DIR, "node_modules", "express")):
            raise unittest.SkipTest("backend node_modules not installed")
        cls.proc = subprocess.Popen(["node", "-e", SERVER_JS], cwd=BACKEND_DIR,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 15
        ready = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}/", timeout=2) as r:
                    r.read()
                    ready = True
                    break
            except Exception:
                # Server prints smoke-ready on stdout once listening; also probe ack route
                try:
                    _post("/1/ack", {"notes": "probe"})
                    ready = True
                    break
                except Exception:
                    time.sleep(0.3)
        if not ready:
            cls.proc.terminate()
            raise unittest.SkipTest("could not start smoke backend on :3998")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.proc.terminate()
            cls.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            db = sqlite3.connect(DB_PATH)
            db.execute("DELETE FROM attack_events WHERE frame_id LIKE 'smoke-%'")
            db.commit()
            db.close()
        except Exception:
            pass

    def test_post_without_hmac_is_401(self):
        status, _ = _post("/", {"confidence": 0.9, "frame_id": "smoke-nohmac-001"})
        self.assertEqual(status, 401)

    def test_post_with_valid_hmac_is_201(self):
        p = {"confidence": 0.9, "frame_id": "smoke-hmac-ok-001",
             "frame_path": "", "timestamp": "2026-08-19T00:00:00+00:00"}
        p["hmac_sig"] = _sig(p)
        status, _ = _post("/", p)
        self.assertEqual(status, 201)

    def test_ack_without_jwt_is_401(self):
        status, _ = _post("/1/ack", {"notes": "x"})
        self.assertEqual(status, 401)


if __name__ == "__main__":
    unittest.main()
