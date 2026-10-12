"""
inference/detector.py
YOLOv8n object detector using OpenCV VideoCapture or Image File / Folder stream.
Supports:
- Live USB webcam / RTSP IP camera feed (via integer index or URL)
- Static image file (e.g., sample.jpg)
- Folder of test images (e.g., data/apde/clean)
- Video file (e.g., sample.mp4)
- Synthetic frame generator ("dummy")
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Union
import uuid
import logging
import os
import glob

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Single detected bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float

    def to_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop and return the ROI from a frame."""
        h, w = frame.shape[:2]
        x1 = max(0, self.x1)
        y1 = max(0, self.y1)
        x2 = min(w, self.x2)
        y2 = min(h, self.y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((128, 128, 3), dtype=np.uint8)
        roi = frame[y1:y2, x1:x2]
        return cv2.resize(roi, (128, 128), interpolation=cv2.INTER_LINEAR)

    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class FrameBundle:
    """Output of one detection cycle.

    NOTE: rgb_frame is OpenCV BGR (despite the name) to avoid extra converts.
    SDNL build_4ch_tensor() expects BGR; FrameStore.save_frame() expects BGR.
    """
    rgb_frame: np.ndarray
    detections: List[BBox] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def is_empty(self) -> bool:
        return len(self.detections) == 0


# COCO classes relevant for infrastructure security & physical attack surfaces
RELEVANT_CLASSES = {
    "person", "car", "motorcycle", "bus", "truck",
    "cell phone", "book", "backpack", "handbag", "laptop", "suitcase"
}


def letterbox(frame: np.ndarray, size: tuple) -> np.ndarray:
    """Resize preserving aspect ratio, pad with neutral gray (YOLO-style)."""
    tw, th = size
    h, w = frame.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(w * s), int(h * s)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((th, tw, 3), 114, dtype=np.uint8)
    out[(th - nh) // 2:(th - nh) // 2 + nh, (tw - nw) // 2:(tw - nw) // 2 + nw] = resized
    return out


class YOLODetector:
    """
    Runs YOLOv8n on frames from live webcam, static image, folder of images, video file, or synthetic frames.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        camera_source: Union[int, str] = 0,
        use_webcam: bool = False,
        conf_threshold: float = 0.5,
        frame_size: tuple = (640, 640),
        imgsz: int = 416,
        detect_every: int = 1,
    ):
        from ultralytics import YOLO

        logger.info(f"Loading YOLOv8n from {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.frame_size = frame_size
        self.imgsz = imgsz
        self.detect_every = max(1, detect_every)
        self._frame_idx = 0
        self._last_detections: List[BBox] = []
        self._cap: Optional[cv2.VideoCapture] = None
        self.use_webcam = use_webcam
        
        if use_webcam and not isinstance(camera_source, int):
            self.camera_source = 0
        else:
            self.camera_source = camera_source

        # Internal state for folder/image modes
        self._mode = "stream"  # "stream", "image", "folder", "dummy"
        self._image_paths = []
        self._image_idx = 0
        self._static_frame = None

        self._connect()

    def _connect(self):
        # 1. Dummy simulation mode
        if self.camera_source == "dummy":
            logger.info("Initializing YOLODetector in DUMMY simulation mode (generating synthetic frames)")
            self._mode = "dummy"
            self._dummy_counter = 0
            return

        # 2. Folder of images
        if isinstance(self.camera_source, str) and os.path.isdir(self.camera_source):
            logger.info(f"Initializing YOLODetector in FOLDER mode: '{self.camera_source}'")
            self._mode = "folder"
            exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
            self._image_paths = []
            for ext in exts:
                self._image_paths.extend(glob.glob(os.path.join(self.camera_source, "**", ext), recursive=True))
            self._image_paths.sort()
            logger.info(f"Found {len(self._image_paths)} test images in directory.")
            if not self._image_paths:
                logger.warning("No images found in directory. Falling back to DUMMY mode.")
                self._mode = "dummy"
                self._dummy_counter = 0
            return

        # 3. Single static image file
        if isinstance(self.camera_source, str) and os.path.isfile(self.camera_source) and self.camera_source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            logger.info(f"Initializing YOLODetector in STATIC IMAGE mode: '{self.camera_source}'")
            self._mode = "image"
            self._static_frame = cv2.imread(self.camera_source)
            if self._static_frame is None:
                logger.warning(f"Could not read image file {self.camera_source}. Falling back to DUMMY mode.")
                self._mode = "dummy"
                self._dummy_counter = 0
            return

        # 4. Webcam or Video file stream via VideoCapture
        logger.info(f"Opening camera / video source: {self.camera_source}")
        self._mode = "stream"
        self._cap = cv2.VideoCapture(self.camera_source)
        if not self._cap.isOpened():
            logger.warning(f"Could not open VideoCapture source {self.camera_source}. Falling back to DUMMY simulation mode.")
            self._mode = "dummy"
            self._dummy_counter = 0
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])

    def get_frame(self) -> Optional[FrameBundle]:
        """
        Capture or load one frame and run YOLOv8n detection.
        Returns FrameBundle or None.
        """
        frame = None

        # Mode A: Dummy synthetic frame
        if self._mode == "dummy":
            self._dummy_counter += 1
            frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (540, 540), (20, 20, 25), -1)
            cv2.putText(frame, "GATE FEED - SIMULATED", (150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
            
            x = int(320 + 150 * np.sin(self._dummy_counter * 0.1))
            y = int(320 + 100 * np.cos(self._dummy_counter * 0.15))
            
            cv2.rectangle(frame, (x - 40, y - 80), (x + 40, y + 80), (120, 80, 50), 2)
            cv2.circle(frame, (x, y - 50), 20, (150, 150, 180), -1)
            
            # Periodic synthetic patch trigger
            if (self._dummy_counter % 50) > 30:
                noise = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
                frame[y - 20:y + 20, x - 20:x + 20] = noise
                cv2.rectangle(frame, (x - 21, y - 21), (x + 21, y + 21), (0, 0, 255), 1)

            detections = [
                BBox(x1=x - 40, y1=y - 80, x2=x + 40, y2=y + 80, label="person", confidence=0.88)
            ]
            time.sleep(0.05) # ~20 FPS sleep
            return FrameBundle(rgb_frame=frame, detections=detections)

        # Mode B: Single Static Image
        elif self._mode == "image":
            frame = self._static_frame.copy()
            time.sleep(0.05)

        # Mode C: Folder of Images
        elif self._mode == "folder":
            if not self._image_paths:
                return None
            img_path = self._image_paths[self._image_idx]
            self._image_idx = (self._image_idx + 1) % len(self._image_paths)
            frame = cv2.imread(img_path)
            time.sleep(0.05)

        # Mode D: Video or Camera Stream
        elif self._mode == "stream":
            if self._cap is None or not self._cap.isOpened():
                self._connect()
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Rewind if video stream ends
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return None

        if frame is None:
            return None

        # Letterbox to frame_size (pad, don't stretch): portrait/phone frames
        # squished to square distort texture frequencies and collapse SDNL
        # scores (measured 0.85 -> 0.10 on the same patch). YOLO letterboxes
        # internally anyway, so detection is unaffected.
        if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
            frame = letterbox(frame, self.frame_size)

        # Frame skipping: reuse last boxes on skipped frames so the live view
        # stays smooth while YOLO runs at 1/detect_every rate (default every frame).
        self._frame_idx += 1
        if (self.detect_every > 1 and (self._frame_idx - 1) % self.detect_every != 0
                and self._last_detections):
            return FrameBundle(rgb_frame=frame, detections=self._last_detections)

        # Run YOLO detection with optimized resolution on CPU
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf_threshold, verbose=False)
        detections: List[BBox] = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                if label not in RELEVANT_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2,
                                       label=label, confidence=conf))

        # If no YOLO bbox found, evaluate candidate ROI
        if len(detections) == 0:
            h, w = frame.shape[:2]
            if self._mode in ["image", "folder"] and min(h, w) <= 256:
                # Standalone test crops ARE the ROI: evaluate the whole image
                detections.append(BBox(x1=0, y1=0, x2=w, y2=h,
                                       label="test_image", confidence=0.99))
            else:
                # Full photos / live frames: TWO center fallback boxes. A big
                # patch blinds YOLO person detection, so the proposal must
                # cover both the diffuse case (70%) and the concentrated case
                # (40%): the worker scores every box and takes the max.
                detections.append(BBox(x1=int(w * 0.15), y1=int(h * 0.15),
                                       x2=int(w * 0.85), y2=int(h * 0.85),
                                       label="camera_field", confidence=0.85))
                detections.append(BBox(x1=int(w * 0.30), y1=int(h * 0.30),
                                       x2=int(w * 0.70), y2=int(h * 0.70),
                                       label="camera_field_tight", confidence=0.85))

        self._last_detections = detections
        return FrameBundle(rgb_frame=frame, detections=detections)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Detector released")
