"""
inference/mqtt_client.py
Paho MQTT wrapper for Smart Gate inference pipeline with Zero-Trust HMAC Attestation.
Publishes attack alerts and heartbeats with HMAC-SHA256 signature; subscribes to override commands.
"""

import json
import logging
import threading
import time
import hmac
import hashlib
import os
from datetime import datetime, timezone
from typing import Callable, Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# Shared secret key for Zero-Trust Edge Attestation (HMAC-SHA256)
HMAC_SHARED_SECRET = os.getenv("HMAC_SHARED_SECRET", "smartgate_zero_trust_hmac_secret_2026").encode("utf-8")


def _canonical_json(payload_dict: dict) -> str:
    """Deterministic canonical form shared by Python, JS, and ESP32.

    Sorted keys, compact separators, no spaces.
    Strings: JSON-quoted. Bool: true/false. Int: decimal.
    Float: fixed %.4f then trim trailing zeros (0.9400 -> 0.94, 20.0000 -> 20).
    This avoids Python `0.0` vs JS `0` vs ESP32 `0.0000` mismatches.
    """
    parts = []
    for k in sorted(payload_dict.keys()):
        if k == "hmac_sig":
            continue
        v = payload_dict[k]
        if isinstance(v, bool):
            s = "true" if v else "false"
        elif isinstance(v, int):
            s = str(v)
        elif isinstance(v, float):
            s = f"{v:.4f}"
            s = s.rstrip("0").rstrip(".")
            if s in ("", "-"):
                s = "0"
            if s == "-0":
                s = "0"
        elif isinstance(v, str):
            s = json.dumps(v)
        elif v is None:
            s = "null"
        else:
            s = json.dumps(v, separators=(",", ":"))
        parts.append(f'"{k}":{s}')
    return "{" + ",".join(parts) + "}"


def generate_payload_signature(payload_dict: dict) -> str:
    """Generates HMAC-SHA256 signature over canonical JSON payload."""
    canonical = _canonical_json(payload_dict)
    return hmac.new(HMAC_SHARED_SECRET, canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_payload_signature(payload_dict: dict) -> bool:
    """Verifies HMAC-SHA256 signature on incoming MQTT message."""
    received_sig = payload_dict.get("hmac_sig", "")
    if not received_sig:
        return False
    expected_sig = generate_payload_signature(payload_dict)
    return hmac.compare_digest(received_sig, expected_sig)


class MQTTClient:
    """
    Thread-safe MQTT client for the inference pipeline with Zero-Trust HMAC Attestation.
    Auto-reconnects on disconnect. QoS 1 for safety-critical messages.
    """

    TOPICS = {
        "attack_alert": "smartgate/attack_alert",
        "attack_ack":   "smartgate/attack_ack",
        "override":     "smartgate/override",
        "heartbeat":    "smartgate/heartbeat",
        "status":       "smartgate/status",
    }

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        keepalive: int = 60,
        override_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.keepalive = keepalive
        self._override_callback = override_callback
        self._connected = False
        self._lock = threading.Lock()

        self._client = mqtt.Client(
            client_id="smartgate_pipeline",
            protocol=mqtt.MQTTv311,
            clean_session=True,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def connect(self):
        """Connect to broker and start background network loop."""
        logger.info(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}")
        try:
            self._client.connect(self.broker_host, self.broker_port, self.keepalive)
            self._client.loop_start()
        except Exception as e:
            logger.warning(f"MQTT broker connection unavailable: {e}. Pipeline operating in standalone HTTP dashboard mode.")

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT client disconnected")

    def publish_alert(self, confidence: float, frame_id: str, frame_path: str = "") -> bool:
        """
        Publish an adversarial patch attack alert with HMAC-SHA256 signature.
        QoS 1 — guaranteed delivery to ESP32 and dashboard.
        """
        raw_payload = {
            "confidence": round(confidence, 4),
            "frame_id": frame_id,
            "frame_path": frame_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        raw_payload["hmac_sig"] = generate_payload_signature(raw_payload)
        payload = json.dumps(raw_payload)

        result = self._client.publish(
            self.TOPICS["attack_alert"], payload, qos=1, retain=False
        )
        ok = result.rc == mqtt.MQTT_ERR_SUCCESS
        if ok:
            logger.info(f"Attested attack alert published (conf={confidence:.3f}, frame={frame_id})")
        else:
            logger.error(f"Failed to publish alert: rc={result.rc}")
        return ok

    def publish_heartbeat(self, fps: float, sdnl_latency_ms: float) -> bool:
        """
        Publish authenticated heartbeat every 5 seconds. QoS 0.
        """
        raw_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fps": round(fps, 1),
            "sdnl_latency_ms": round(sdnl_latency_ms, 1),
        }
        raw_payload["hmac_sig"] = generate_payload_signature(raw_payload)
        payload = json.dumps(raw_payload)

        result = self._client.publish(
            self.TOPICS["heartbeat"], payload, qos=0, retain=False
        )
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def is_connected(self) -> bool:
        return self._connected

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected to broker")
            client.subscribe(self.TOPICS["override"], qos=1)
            client.subscribe(self.TOPICS["attack_ack"], qos=1)
            logger.info(f"Subscribed to: {self.TOPICS['override']}, {self.TOPICS['attack_ack']}")
        else:
            logger.error(f"MQTT connection failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnect (rc={rc}), auto-reconnect active")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            logger.warning("Failed to decode MQTT JSON payload")
            return

        if isinstance(payload, dict) and not verify_payload_signature(payload):
            logger.warning(f"[ZERO-TRUST REJECT] Message on {topic} failed HMAC-SHA256 signature attestation!")
            return

        if topic == self.TOPICS["override"] and self._override_callback:
            logger.info(f"Attested override command received: {payload}")
            self._override_callback(payload)
        elif topic == self.TOPICS["attack_ack"]:
            logger.info(f"ESP32 ACK received: {payload}")
