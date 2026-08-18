"""
inference/mqtt_client.py
Paho MQTT wrapper for InfraEdge-Shield inference pipeline.
Publishes attack alerts and heartbeats; subscribes to override commands.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTClient:
    """
    Thread-safe MQTT client for the inference pipeline.
    Auto-reconnects on disconnect. QoS 1 for safety-critical messages.
    """

    TOPICS = {
        "attack_alert": "infraedge/attack_alert",
        "attack_ack":   "infraedge/attack_ack",
        "override":     "infraedge/override",
        "heartbeat":    "infraedge/heartbeat",
        "status":       "infraedge/status",
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
            client_id="infraedge_pipeline",
            protocol=mqtt.MQTTv311,
            clean_session=True,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def connect(self):
        """Connect to broker and start background network loop."""
        logger.info(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}")
        self._client.connect(self.broker_host, self.broker_port, self.keepalive)
        self._client.loop_start()

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT client disconnected")

    def publish_alert(self, confidence: float, frame_id: str, frame_path: str = "") -> bool:
        """
        Publish an adversarial patch attack alert.
        QoS 1 — guaranteed delivery to ESP32 and dashboard.
        """
        payload = json.dumps({
            "confidence": round(confidence, 4),
            "frame_id": frame_id,
            "frame_path": frame_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        result = self._client.publish(
            self.TOPICS["attack_alert"], payload, qos=1, retain=False
        )
        ok = result.rc == mqtt.MQTT_ERR_SUCCESS
        if ok:
            logger.info(f"Attack alert published (conf={confidence:.3f}, frame={frame_id})")
        else:
            logger.error(f"Failed to publish alert: rc={result.rc}")
        return ok

    def publish_heartbeat(self, fps: float, sdnl_latency_ms: float) -> bool:
        """
        Publish heartbeat every 5 seconds. QoS 0 (fire-and-forget).
        ESP32 watchdog resets on receipt; auto-locks if silent for 15s.
        """
        payload = json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fps": round(fps, 1),
            "sdnl_latency_ms": round(sdnl_latency_ms, 1),
        })
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
            # Subscribe to override commands from dashboard
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
            payload = msg.payload.decode()

        if topic == self.TOPICS["override"] and self._override_callback:
            logger.info(f"Override command received: {payload}")
            self._override_callback(payload)
        elif topic == self.TOPICS["attack_ack"]:
            logger.info(f"ESP32 ACK received: {payload}")
