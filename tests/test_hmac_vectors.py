"""HMAC known-answer vectors — guards the Python/JS/ESP32 canonical form."""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from mqtt_client import generate_payload_signature, verify_payload_signature


class TestHmacVectors(unittest.TestCase):
    def test_attack_alert_vector(self):
        p = {"confidence": 0.94, "frame_id": "abc123",
             "frame_path": "/x.jpg", "timestamp": "2026-08-19T00:00:00+00:00"}
        # Pin the first 16 hex chars (stable across PY/JS/ESP32)
        self.assertEqual(generate_payload_signature(p)[:16], "7a1969d7a4c55c24")
        p["hmac_sig"] = generate_payload_signature(p)
        self.assertTrue(verify_payload_signature(p))

    def test_whole_float_vector(self):
        p = {"timestamp": "2026-08-19T00:00:00+00:00", "fps": 20.0, "sdnl_latency_ms": 0.0}
        self.assertEqual(generate_payload_signature(p)[:16], "46538bc8b029bc2d")
        p["hmac_sig"] = generate_payload_signature(p)
        self.assertTrue(verify_payload_signature(p))

    def test_tamper_rejected(self):
        p = {"confidence": 0.94, "frame_id": "abc123",
             "frame_path": "/x.jpg", "timestamp": "2026-08-19T00:00:00+00:00"}
        p["hmac_sig"] = generate_payload_signature(p)
        p["confidence"] = 0.11
        self.assertFalse(verify_payload_signature(p))

    def test_missing_sig_rejected(self):
        self.assertFalse(verify_payload_signature({"confidence": 0.9}))


if __name__ == "__main__":
    unittest.main()
