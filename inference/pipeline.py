"""
inference/pipeline.py
InfraEdge-Shield dual-model inference pipeline.

3-thread design:
  Thread 1 (Frame Reader)    — captures frames / reads test images + runs YOLOv8n
  Thread 2 (SDNL Worker)     — runs SDNL 4-channel classifier on each ROI
  Thread 3 (Alert Publisher) — saves frames, publishes MQTT, POSTs to backend

Run commands:
  python inference/pipeline.py --dry-run                           # Default dummy/simulation mode
  python inference/pipeline.py --image data/apde/attack/img_0.jpg  # Static image mode
  python inference/pipeline.py --image-dir data/apde/attack        # Image folder mode
  python inference/pipeline.py --video test_video.mp4              # Video file mode
  python inference/pipeline.py --use-webcam                        # Live webcam mode (toggle)
"""

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import cv2
import requests
import yaml

from detector import FrameBundle, YOLODetector
from frame_store import FrameStore
from mqtt_client import MQTTClient
from sdnl import SDNLClassifier

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    camera_source: object = "dummy"    # int (USB webcam index), str (file/folder path), or "dummy"
    use_webcam: bool = False
    yolo_model: str = "yolov8n.pt"
    sdnl_model: str = "models/sdnl/sdnl_v1.onnx"
    sdnl_threshold: float = 0.85
    broker_host: str = "localhost"
    broker_port: int = 1883
    backend_url: str = "http://localhost:3000"
    frames_dir: str = "data/frames"
    heartbeat_interval_s: int = 5
    max_frames: Optional[int] = None
    dry_run: bool = False              # skip MQTT + backend POST

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        inf = raw.get("inference", {})
        mq = raw.get("mqtt", {})
        be = raw.get("backend", {})
        st = raw.get("storage", {})

        cam_url = inf.get("camera_url", "")
        cfg.camera_source = cam_url if cam_url else inf.get("camera_index", "dummy")
        cfg.yolo_model = inf.get("yolo_model", cfg.yolo_model)
        cfg.sdnl_model = inf.get("sdnl_model", cfg.sdnl_model)
        cfg.sdnl_threshold = inf.get("sdnl_threshold", cfg.sdnl_threshold)
        cfg.broker_host = mq.get("broker_host", cfg.broker_host)
        cfg.broker_port = mq.get("broker_port", cfg.broker_port)
        cfg.backend_url = f"http://{be.get('host','localhost')}:{be.get('port',3000)}"
        cfg.frames_dir = st.get("frames_dir", cfg.frames_dir)
        cfg.heartbeat_interval_s = raw.get("esp32", {}).get("heartbeat_interval_s", 5)
        return cfg


# ── Alert payload passed between threads ──────────────────────────────────────
@dataclass
class AlertPayload:
    bundle: FrameBundle
    confidence: float


# ── Pipeline ──────────────────────────────────────────────────────────────────
class DualModelPipeline:
    """
    Three-thread detection pipeline.
    Bounded queues provide natural backpressure to prevent memory buildup.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._running = False

        # Bounded queues
        self._roi_queue: queue.Queue = queue.Queue(maxsize=10)
        self._alert_queue: queue.Queue = queue.Queue(maxsize=5)

        # Metrics
        self._fps = 0.0
        self._sdnl_latency_ms = 0.0
        self._frame_count = 0
        self._attack_count = 0
        self._t_start = 0.0

        # Components
        self._detector: Optional[YOLODetector] = None
        self._sdnl: Optional[SDNLClassifier] = None
        self._mqtt: Optional[MQTTClient] = None
        self._store: Optional[FrameStore] = None

    def start(self):
        logger.info("=== InfraEdge-Shield Dual-Model Pipeline Starting ===")
        self._t_start = time.time()
        self._running = True

        # Initialize components
        logger.info("Loading YOLOv8n detector...")
        self._detector = YOLODetector(
            model_path=self.config.yolo_model,
            camera_source=self.config.camera_source,
            use_webcam=self.config.use_webcam,
            conf_threshold=0.5,
        )

        logger.info("Loading SDNL classifier...")
        self._sdnl = SDNLClassifier(
            model_path=self.config.sdnl_model,
            threshold=self.config.sdnl_threshold,
        )

        self._store = FrameStore(frames_dir=self.config.frames_dir)

        if not self.config.dry_run:
            logger.info("Connecting to MQTT broker...")
            self._mqtt = MQTTClient(
                broker_host=self.config.broker_host,
                broker_port=self.config.broker_port,
            )
            self._mqtt.connect()
            time.sleep(0.5)
        else:
            logger.info("DRY RUN mode — MQTT and backend POST disabled")

        # Start threads
        threads = [
            threading.Thread(target=self._frame_reader, name="T1-FrameReader", daemon=True),
            threading.Thread(target=self._sdnl_worker,  name="T2-SDNLWorker",  daemon=True),
            threading.Thread(target=self._alert_publisher, name="T3-AlertPublisher", daemon=True),
        ]
        for t in threads:
            t.start()
        logger.info("All 3 threads operational. Monitoring stream...")

        # Main thread: heartbeat and max frames check loop
        try:
            while self._running:
                time.sleep(1.0)
                self._send_heartbeat()
                self._log_stats()
                
                if self.config.max_frames and self._frame_count >= self.config.max_frames:
                    logger.info(f"Reached max frames limit ({self.config.max_frames}). Stopping pipeline.")
                    break
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self):
        logger.info("Stopping pipeline...")
        self._running = False
        if self._detector:
            self._detector.release()
        if self._mqtt:
            self._mqtt.disconnect()
        uptime = time.time() - self._t_start if self._t_start else 0.0
        logger.info("Pipeline stopped. Final Stats | Processed Frames: %d | Attacks Detected: %d | Uptime: %.1fs",
                    self._frame_count, self._attack_count, uptime)

    # ── Thread 1: Frame Reader ────────────────────────────────────────────────
    def _frame_reader(self):
        fps_window_start = time.time()
        fps_count = 0

        while self._running:
            bundle = self._detector.get_frame()
            if bundle is None:
                time.sleep(0.05)
                continue

            self._frame_count += 1
            fps_count += 1

            # Update FPS every 2 seconds
            elapsed = time.time() - fps_window_start
            if elapsed >= 2.0:
                self._fps = fps_count / elapsed
                fps_count = 0
                fps_window_start = time.time()

            if bundle.is_empty():
                continue

            # Push to SDNL worker queue (drop oldest if full)
            try:
                self._roi_queue.put_nowait(bundle)
            except queue.Full:
                try:
                    self._roi_queue.get_nowait()  # Drop oldest
                    self._roi_queue.put_nowait(bundle)
                except queue.Empty:
                    pass

    # ── Thread 2: SDNL Worker ────────────────────────────────────────────────
    def _sdnl_worker(self):
        while self._running:
            try:
                bundle: FrameBundle = self._roi_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            max_confidence = 0.0

            for bbox in bundle.detections:
                roi = bbox.to_roi(bundle.rgb_frame)
                label, conf = self._sdnl.infer(roi)
                self._sdnl_latency_ms = self._sdnl.get_latency_ms()

                if label == "ATTACK" and conf > max_confidence:
                    max_confidence = conf

            if max_confidence >= self.config.sdnl_threshold:
                self._attack_count += 1
                alert = AlertPayload(bundle=bundle, confidence=max_confidence)
                try:
                    self._alert_queue.put_nowait(alert)
                except queue.Full:
                    logger.warning("Alert queue full — dropping alert")

    # ── Thread 3: Alert Publisher ─────────────────────────────────────────────
    def _alert_publisher(self):
        while self._running:
            try:
                alert: AlertPayload = self._alert_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_id = alert.bundle.frame_id

            # 1. Save flagged frame
            frame_path = self._store.save_frame(alert.bundle.rgb_frame, frame_id)

            # 2. Publish MQTT alert (QoS 1)
            if self._mqtt and self._mqtt.is_connected():
                self._mqtt.publish_alert(
                    confidence=alert.confidence,
                    frame_id=frame_id,
                    frame_path=frame_path,
                )

            # 3. POST to Node.js backend
            if not self.config.dry_run:
                try:
                    requests.post(
                        f"{self.config.backend_url}/api/events",
                        json={
                            "confidence": alert.confidence,
                            "frame_path": frame_path,
                            "frame_id": frame_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        timeout=2.0,
                    )
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Backend POST failed: {e}")

            if self.config.dry_run:
                logger.info(
                    f"[ALERT DETECTED] ATTACK Patch Identified | conf={alert.confidence:.3f} | "
                    f"frame_id={frame_id} | saved={frame_path}"
                )

    # ── Heartbeat ─────────────────────────────────────────────────────────────
    def _send_heartbeat(self):
        if self._mqtt and self._mqtt.is_connected():
            self._mqtt.publish_heartbeat(
                fps=self._fps,
                sdnl_latency_ms=self._sdnl_latency_ms,
            )

    def _log_stats(self):
        uptime = time.time() - self._t_start
        logger.info(
            f"Pipeline Telemetry | FPS={self._fps:.1f} | SDNL Latency={self._sdnl_latency_ms:.1f}ms | "
            f"Frames Processed={self._frame_count} | Attacks Flagged={self._attack_count} | "
            f"Uptime={uptime:.0f}s"
        )


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="InfraEdge-Shield Inference Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config file")
    
    # Input source options (Bypass Camera)
    parser.add_argument("--use-webcam", action="store_true", help="Toggle live USB webcam feed (index 0)")
    parser.add_argument("--image", type=str, default=None, help="Path to single static image file")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to folder of test images")
    parser.add_argument("--video", type=str, default=None, help="Path to sample video file")
    parser.add_argument("--dummy", action="store_true", help="Use synthetic simulation generator (default fallback)")
    
    parser.add_argument("--threshold", type=float, default=None, help="SDNL detection threshold (default 0.85)")
    parser.add_argument("--max-frames", type=int, default=None, help="Auto-stop pipeline after processing N frames")
    parser.add_argument("--dry-run", action="store_true", help="Disable MQTT broker and backend POST (dry-run mode)")
    args = parser.parse_args()

    # Load config
    try:
        config = PipelineConfig.from_yaml(args.config)
    except FileNotFoundError:
        logger.warning(f"Config not found at '{args.config}', using defaults")
        config = PipelineConfig()

    # Resolve input source hierarchy
    if args.use_webcam:
        config.use_webcam = True
        config.camera_source = 0
    elif args.image:
        config.camera_source = args.image
    elif args.image_dir:
        config.camera_source = args.image_dir
    elif args.video:
        config.camera_source = args.video
    elif args.dummy:
        config.camera_source = "dummy"

    if args.threshold is not None:
        config.sdnl_threshold = args.threshold
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    if args.dry_run:
        config.dry_run = True

    logger.info(f"Input Source: '{config.camera_source}' (Webcam Toggle: {config.use_webcam}) | "
                f"SDNL Threshold: {config.sdnl_threshold} | Dry-run: {config.dry_run}")

    pipeline = DualModelPipeline(config)
    signal.signal(signal.SIGTERM, lambda s, f: pipeline.stop())
    pipeline.start()


if __name__ == "__main__":
    main()
