"""Detector speed knobs: imgsz plumbing + detect_every box reuse (stubbed YOLO)."""
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.detector import YOLODetector
from inference.pipeline import PipelineConfig


class _EmptyResult:
    boxes = []


class _CountingModel:
    def __init__(self):
        self.calls = 0
        self.names = {}
    def __call__(self, *a, **k):
        self.calls += 1
        return [_EmptyResult()]


class TestDetectEvery(unittest.TestCase):
    def _detector(self, detect_every):
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, np.full((64, 64, 3), 128, np.uint8))
        self.addCleanup(os.remove, path)
        d = YOLODetector(camera_source=path, imgsz=320, detect_every=detect_every)
        d.model = _CountingModel()
        return d

    def test_every_1_runs_yolo_each_frame(self):
        d = self._detector(detect_every=1)
        for _ in range(3):
            d.get_frame()
        self.assertEqual(d.model.calls, 3)

    def test_every_2_reuses_boxes(self):
        d = self._detector(detect_every=2)
        b1 = d.get_frame()  # runs YOLO (idx 1)
        b2 = d.get_frame()  # reuses (idx 2)
        b3 = d.get_frame()  # runs YOLO (idx 3)
        self.assertEqual(d.model.calls, 2)
        boxes = lambda b: [(x.x1, x.y1, x.x2, x.y2) for x in b.detections]
        self.assertEqual(boxes(b1), boxes(b2))
        self.assertEqual(boxes(b2), boxes(b3))

    def test_config_plumbing(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.yolo_imgsz, 416)
        self.assertEqual(cfg.detect_every, 1)


if __name__ == "__main__":
    unittest.main()
