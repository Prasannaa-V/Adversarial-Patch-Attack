"""
inference/pipeline.py
InfraEdge-Shield dual-model inference pipeline.

3-thread design:
  Thread 1 (Frame Reader)    — captures frames / reads test images + runs YOLOv8n
  Thread 2 (SDNL Worker)     — runs SDNL 4-channel classifier on each ROI
  Thread 3 (Alert Publisher) — saves frames, publishes MQTT, POSTs to backend

Run commands:
  python inference/pipeline.py --dry-run                           # Default dummy/simulation mode
  python inference/pipeline.py --image data/crops/attack/<file>.jpg  # Static image mode
  python inference/pipeline.py --image-dir data/crops/attack        # Image folder mode
  python inference/pipeline.py --video test_video.mp4              # Video file mode
  python inference/pipeline.py --use-webcam                        # Live webcam mode (toggle)
"""

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import requests
import yaml

try:
    from detector import FrameBundle, YOLODetector
    from frame_store import FrameStore
    from mqtt_client import MQTTClient, generate_payload_signature
    from sdnl import SDNLClassifier
except ImportError:
    from inference.detector import FrameBundle, YOLODetector
    from inference.frame_store import FrameStore
    from inference.mqtt_client import MQTTClient, generate_payload_signature
    from inference.sdnl import SDNLClassifier

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
    sdnl_threshold: float = 0.30
    broker_host: str = "localhost"
    broker_port: int = 1883
    backend_url: str = "http://localhost:3000"
    frames_dir: str = "data/frames"
    heartbeat_interval_s: int = 5
    max_frames: Optional[int] = None
    dry_run: bool = False              # skip MQTT + backend POST
    demo_attack_every: int = 0         # 0=disabled; N>0 forces conf=0.95 attack every Nth frame (demo)
    demo_confidence: float = 0.95
    alert_cooldown_s: float = 15.0     # min seconds between stored alerts (one patch = one incident)
    qr_suppress: bool = True           # suppress decodable QR structured codes (badges/passes)
    roi_tiles: int = 1                 # EXPERIMENTAL: tiles×tiles grid max (measured: no gain + 70-110ms). 1 = whole ROI
    score_log: Optional[str] = None    # CSV path for per-tile score calibration (None = disabled)

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
        cfg.alert_cooldown_s = float(inf.get("alert_cooldown_s", cfg.alert_cooldown_s))
        cfg.qr_suppress = bool(inf.get("qr_suppress", cfg.qr_suppress))
        cfg.roi_tiles = int(inf.get("roi_tiles", cfg.roi_tiles))
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

        # Metrics & Thread Safety Lock
        self._metrics_lock = threading.Lock()
        self._fps = 0.0
        self._sdnl_latency_ms = 0.0
        self._frame_count = 0
        self._attack_count = 0
        self._demo_counter = 0
        self._last_alert_ts = 0.0
        self._t_start = 0.0
        self._live_frame_path = os.path.join(config.frames_dir, "live_frame.jpg")

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
            qr_suppress=self.config.qr_suppress,
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
        last_heartbeat_time = time.time()
        last_cleanup_time = time.time()

        try:
            while self._running:
                time.sleep(0.5)
                now = time.time()

                # Send heartbeat per config cadence (H3 fix)
                if now - last_heartbeat_time >= self.config.heartbeat_interval_s:
                    self._send_heartbeat()
                    self._log_stats()
                    last_heartbeat_time = now

                # Cleanup old frames every 10 minutes (M2 fix)
                if now - last_cleanup_time >= 600:
                    if self._store:
                        self._store.cleanup_old_frames(max_days=7)
                    last_cleanup_time = now
                
                with self._metrics_lock:
                    current_frames = self._frame_count

                if self.config.max_frames and current_frames >= self.config.max_frames:
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
        with self._metrics_lock:
            fc, ac = self._frame_count, self._attack_count
        logger.info("Pipeline stopped. Final Stats | Processed Frames: %d | Attacks Detected: %d | Uptime: %.1fs",
                    fc, ac, uptime)

    # ── Thread 1: Frame Reader ────────────────────────────────────────────────
    def _frame_reader(self):
        fps_window_start = time.time()
        fps_count = 0

        while self._running:
            # Hard stop: prevents max_frames overshoot from queued frames
            with self._metrics_lock:
                if self.config.max_frames and self._frame_count >= self.config.max_frames:
                    self._running = False
                    break
            bundle = self._detector.get_frame()
            if bundle is None:
                time.sleep(0.05)
                continue

            # Pace image folder playback for presentation demos (1s per image).
            # Live URL streams (http/rtsp) must NOT sleep — same as webcam.
            src = self.config.camera_source
            if (not self.config.use_webcam and isinstance(src, str) and src != "dummy"
                    and "://" not in src):
                time.sleep(1.0)

            with self._metrics_lock:
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

    def should_publish_attack(self, confidence: float, now: float) -> bool:
        """One patch = one incident: publish only if over threshold AND
        cooldown elapsed since the last published alert. Thread-safe."""
        if confidence < self.config.sdnl_threshold:
            return False
        with self._metrics_lock:
            if (now - self._last_alert_ts) < self.config.alert_cooldown_s:
                return False
            self._last_alert_ts = now
            self._attack_count += 1
        return True

    # ── Thread 2: SDNL Worker ────────────────────────────────────────────────
    @staticmethod
    def split_tiles(roi: np.ndarray, tiles: int):
        """Split an ROI into a tiles×tiles grid (localized patch regions).

        Training saw tight patch crops; live ROIs are whole person boxes where
        a patch is diluted by background. Scoring tiles and taking the max
        restores the train/infer distribution match. tiles=1 disables tiling.
        """
        if tiles <= 1:
            return [roi]
        h, w = roi.shape[:2]
        out = []
        for gy in range(tiles):
            for gx in range(tiles):
                y0, y1 = gy * h // tiles, (gy + 1) * h // tiles
                x0, x1 = gx * w // tiles, (gx + 1) * w // tiles
                if y1 > y0 and x1 > x0:
                    out.append(roi[y0:y1, x0:x1])
        return out or [roi]

    def _sdnl_worker(self):
        scorer = None
        if self.config.score_log:
            fresh = not os.path.exists(self.config.score_log) or os.path.getsize(self.config.score_log) == 0
            scorer = open(self.config.score_log, "a", buffering=1)
            if fresh:
                scorer.write("timestamp,frame_id,bbox,tile,conf,published\n")
        try:
            while self._running:
                try:
                    bundle: FrameBundle = self._roi_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                max_confidence = 0.0
                tile_rows = []  # (bbox_idx, tile_idx, conf) for optional score logging

                for bi, bbox in enumerate(bundle.detections):
                    roi = bbox.to_roi(bundle.rgb_frame)
                    for ti, tile in enumerate(self.split_tiles(roi, self.config.roi_tiles)):
                        label, conf = self._sdnl.infer(tile)
                        with self._metrics_lock:
                            self._sdnl_latency_ms = self._sdnl.get_latency_ms()

                        if label == "ATTACK" and conf > max_confidence:
                            max_confidence = conf
                        tile_rows.append((bi, ti, conf))

                with self._metrics_lock:
                    self._demo_counter += 1
                    demo_n = self.config.demo_attack_every
                    demo_hit = demo_n > 0 and (self._demo_counter % demo_n == 0)
                if demo_hit:
                    max_confidence = max(max_confidence, self.config.demo_confidence)
                    logger.info(f"[DEMO] Forced attack frame #{self._demo_counter} conf={max_confidence:.2f}")

                now = time.time()
                if demo_hit:
                    # Explicit demo intent bypasses cooldown so demos stay lively.
                    with self._metrics_lock:
                        self._last_alert_ts = now
                        self._attack_count += 1
                    publish = True
                else:
                    publish = self.should_publish_attack(max_confidence, now)
                if publish:
                    alert = AlertPayload(bundle=bundle, confidence=max_confidence)
                    try:
                        self._alert_queue.put_nowait(alert)
                    except queue.Full:
                        logger.warning("Alert queue full — dropping alert")

                if scorer is not None:
                    flag = 1 if publish else 0
                    for bi, ti, conf in tile_rows:
                        scorer.write(f"{now:.3f},{bundle.frame_id},{bi},{ti},{conf:.4f},{flag}\n")

                # Write annotated live frame snapshot for smooth dashboard streaming
                self._save_live_frame(bundle, max_confidence)
        finally:
            if scorer is not None:
                scorer.close()

    # ── Live frame writer ─────────────────────────────────────────────────────
    def _save_live_frame(self, bundle: FrameBundle, max_confidence: float):
        """Draw bounding boxes + SDNL label on frame and write to live_frame.jpg.

        NOTE: bundle.rgb_frame is OpenCV BGR (despite the name). All drawing
        uses BGR tuples and the frame is written directly with imwrite.
        """
        try:
            os.makedirs(os.path.dirname(self._live_frame_path), exist_ok=True)
            frame = bundle.rgb_frame.copy()
            is_attack = max_confidence >= self.config.sdnl_threshold
            colour = (68, 68, 239) if is_attack else (129, 185, 16)   # red / green in BGR

            for bbox in bundle.detections:
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Overlay banner (BGR backgrounds)
            label = f"ATTACK  {max_confidence*100:.1f}%" if is_attack else "CLEAN"
            banner_bg = (10, 10, 60) if is_attack else (30, 45, 10)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), banner_bg, -1)
            cv2.putText(frame, f"InfraEdge-Shield  |  {label}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)

            # Timestamp watermark
            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, ts, (frame.shape[1] - 80, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imwrite(self._live_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            logger.debug(f"Live frame write failed: {e}")

    # ── Thread 3: Alert Publisher ─────────────────────────────────────────────
    def _post_alert_async(self, backend_url: str, confidence: float, frame_path: str, frame_id: str):
        try:
            payload = {
                "confidence": round(float(confidence), 4),
                "frame_path": frame_path,
                "frame_id": frame_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            payload["hmac_sig"] = generate_payload_signature(payload)
            requests.post(
                f"{backend_url}/api/events",
                json=payload,
                timeout=2.0,
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Async Backend POST failed: {e}")

    def _alert_publisher(self):
        while self._running:
            try:
                alert: AlertPayload = self._alert_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_id = alert.bundle.frame_id

            # 1. Save flagged frame
            frame_path = self._store.save_frame(alert.bundle.rgb_frame, frame_id)

            # 2. Publish MQTT alert (QoS 1) - Critical path
            if self._mqtt and self._mqtt.is_connected():
                self._mqtt.publish_alert(
                    confidence=alert.confidence,
                    frame_id=frame_id,
                    frame_path=frame_path,
                )

            # 3. Asynchronous non-blocking POST to Node.js backend (H4 fix)
            if not self.config.dry_run:
                threading.Thread(
                    target=self._post_alert_async,
                    args=(self.config.backend_url, alert.confidence, frame_path, frame_id),
                    daemon=True,
                ).start()

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
        # Also post to HTTP backend to keep web dashboard active even without MQTT broker
        try:
            requests.post(
                f"{self.config.backend_url}/api/heartbeat",
                json={"fps": round(self._fps, 1), "sdnl_latency_ms": round(self._sdnl_latency_ms, 2)},
                timeout=1.0,
            )
        except Exception:
            pass

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
    parser.add_argument("--camera-url", type=str, default=None, help="Phone/IP camera stream URL (e.g. http://192.168.1.5:8080/video)")
    parser.add_argument("--image", type=str, default=None, help="Path to single static image file")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to folder of test images")
    parser.add_argument("--video", type=str, default=None, help="Path to sample video file")
    parser.add_argument("--dummy", action="store_true", help="Use synthetic simulation generator (default fallback)")
    
    parser.add_argument("--threshold", type=float, default=None, help="SDNL detection threshold (default 0.30 from config)")
    parser.add_argument("--alert-cooldown", type=float, default=None, help="Min seconds between stored alerts (default 15 from config)")
    parser.add_argument("--roi-tiles", type=int, default=None, help="EXPERIMENTAL: score ROI as N×N tile grid, take max (default 1 = whole ROI)")
    parser.add_argument("--score-log", type=str, default=None, help="Append per-tile scores to CSV for threshold calibration")
    parser.add_argument("--no-qr-suppress", action="store_true", help="Disable QR structured-code suppression")
    parser.add_argument("--max-frames", type=int, default=None, help="Auto-stop pipeline after processing N frames")
    parser.add_argument("--dry-run", action="store_true", help="Disable MQTT broker and backend POST (dry-run mode)")
    parser.add_argument("--demo-attack-every", type=int, default=0, help="Demo mode: force conf=0.95 attack every Nth frame (e.g. 10)")
    args = parser.parse_args()

    # Load config
    try:
        config = PipelineConfig.from_yaml(args.config)
    except FileNotFoundError:
        logger.warning(f"Config not found at '{args.config}', using defaults")
        config = PipelineConfig()

    # Resolve input source hierarchy (camera URL wins, like config.yaml)
    if args.camera_url:
        config.use_webcam = False
        config.camera_source = args.camera_url
    elif args.use_webcam:
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
    if args.alert_cooldown is not None:
        config.alert_cooldown_s = args.alert_cooldown
    if args.roi_tiles is not None:
        config.roi_tiles = args.roi_tiles
    if args.score_log is not None:
        config.score_log = args.score_log
    if args.no_qr_suppress:
        config.qr_suppress = False
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    if args.dry_run:
        config.dry_run = True
    if args.demo_attack_every:
        config.demo_attack_every = args.demo_attack_every

    logger.info(f"Input Source: '{config.camera_source}' (Webcam Toggle: {config.use_webcam}) | "
                f"SDNL Threshold: {config.sdnl_threshold} | Dry-run: {config.dry_run} | "
                f"DemoAttackEvery: {config.demo_attack_every}")

    pipeline = DualModelPipeline(config)
    signal.signal(signal.SIGTERM, lambda s, f: pipeline.stop())
    pipeline.start()


if __name__ == "__main__":
    main()
