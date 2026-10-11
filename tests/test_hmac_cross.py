"""Python ↔ JS HMAC cross-compat via node subprocess (skips if node missing)."""
import json
import os
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from mqtt_client import generate_payload_signature

JS_SNIPPET = r"""
const crypto = require('crypto');
const SECRET = 'smartgate_zero_trust_hmac_secret_2026';
function canon(o){const ks=Object.keys(o).filter(k=>k!=='hmac_sig').sort();return '{'+ks.map(k=>{const v=o[k];let s;if(typeof v==='string')s=JSON.stringify(v);else if(typeof v==='boolean')s=v?'true':'false';else if(typeof v==='number'){if(Number.isInteger(v))s=String(v);else{s=v.toFixed(4).replace(/0+$/,'').replace(/\.$/,'');if(s==='-0'||s==='')s='0';}}else if(v===null||v===undefined)s='null';else s=JSON.stringify(v);return '"'+k+'":'+s;}).join(',')+'}';}
const payload = JSON.parse(process.argv[1]);
process.stdout.write(crypto.createHmac('sha256', SECRET).update(canon(payload)).digest('hex'));
"""


@unittest.skipIf(shutil.which("node") is None, "node not available")
class TestHmacCrossCompat(unittest.TestCase):
    def _js_sig(self, payload):
        r = subprocess.run(["node", "-e", JS_SNIPPET, json.dumps(payload)],
                           capture_output=True, text=True, timeout=15)
        self.assertEqual(r.returncode, 0, msg=r.stderr[:300])
        return r.stdout.strip()

    def test_attack_alert_matches_js(self):
        p = {"confidence": 0.94, "frame_id": "abc123",
             "frame_path": "/x.jpg", "timestamp": "2026-08-19T00:00:00+00:00"}
        self.assertEqual(generate_payload_signature(p), self._js_sig(p))

    def test_heartbeat_matches_js(self):
        p = {"timestamp": "2026-08-19T00:00:00+00:00", "fps": 20.0, "sdnl_latency_ms": 1.6}
        self.assertEqual(generate_payload_signature(p), self._js_sig(p))


if __name__ == "__main__":
    unittest.main()
