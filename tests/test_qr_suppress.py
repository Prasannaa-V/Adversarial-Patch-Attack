"""QR structured-code suppression (mocked detector; no extra deps)."""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
import sdnl
from sdnl import SDNLClassifier, qr_code_present


class TestQrSuppress(unittest.TestCase):
    def test_detector_guards_invalid_input(self):
        self.assertFalse(qr_code_present(None))
        self.assertFalse(qr_code_present(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertFalse(qr_code_present(np.zeros((8, 8, 3), dtype=np.uint8)))

    def test_attack_with_qr_suppressed(self):
        roi = np.full((128, 128, 3), 200, dtype=np.uint8)
        # threshold=0.0 forces the ATTACK path; mocked QR must flip it to CLEAN
        clf = SDNLClassifier(threshold=0.0, qr_suppress=True)
        with patch.object(sdnl, "qr_code_present", return_value=True):
            label, conf = clf.infer(roi)
        self.assertEqual(label, "CLEAN")
        # Same input unsuppressed stays ATTACK with identical score
        clf2 = SDNLClassifier(threshold=0.0, qr_suppress=False)
        with patch.object(sdnl, "qr_code_present", return_value=True):
            label2, conf2 = clf2.infer(roi)
        self.assertEqual(label2, "ATTACK")
        self.assertEqual(conf, conf2)

    def test_no_suppress_when_disabled(self):
        clf = SDNLClassifier(threshold=0.0, qr_suppress=False)  # everything is ATTACK
        roi = np.full((128, 128, 3), 200, dtype=np.uint8)
        with patch.object(sdnl, "qr_code_present", return_value=True):
            label, _ = clf.infer(roi)
        self.assertEqual(label, "ATTACK")

    def test_clean_path_ignores_qr_detector(self):
        clf = SDNLClassifier(threshold=0.70, qr_suppress=True)
        roi = np.full((128, 128, 3), 200, dtype=np.uint8)
        with patch.object(sdnl, "qr_code_present", side_effect=AssertionError("must not run on CLEAN path")):
            label, _ = clf.infer(roi)
        self.assertEqual(label, "CLEAN")

    def test_no_real_attack_decodes_as_qr(self):
        # Decodable-only suppression must never hide a true attack: random
        # swirls essentially never decode. Sampled for speed.
        import glob
        import cv2
        files = sorted(glob.glob("data/crops/attack/*.jpg"))[::15]
        self.assertGreater(len(files), 10)
        for p in files:
            self.assertFalse(qr_code_present(cv2.imread(p)), msg=p)

    def test_pipeline_qr_suppressed_does_not_raise_confidence_or_publish(self):
        """A region scored as a decodable QR must not raise max_confidence above 0
        and must not publish an alert."""
        from detector import FrameBundle, BBox
        from pipeline import DualModelPipeline, PipelineConfig

        cfg = PipelineConfig(sdnl_threshold=0.30, qr_suppress=True, alert_cooldown_s=0.0)
        pipeline = DualModelPipeline(cfg)

        # Mock SDNL classifier returning CLEAN with high confidence (emulating QR suppression)
        mock_sdnl = MagicMock()
        mock_sdnl.infer.return_value = ("CLEAN", 0.85)
        mock_sdnl.get_latency_ms.return_value = 1.0
        pipeline._sdnl = mock_sdnl

        frame = np.full((128, 128, 3), 200, dtype=np.uint8)
        bundle = FrameBundle(
            rgb_frame=frame,
            detections=[BBox(x1=0, y1=0, x2=128, y2=128, label="person", confidence=0.95)]
        )
        pipeline._roi_queue.put(bundle)

        recorded_max_conf = []

        def intercept_save_live_frame(b, max_conf):
            recorded_max_conf.append(max_conf)
            pipeline._running = False

        pipeline._save_live_frame = intercept_save_live_frame
        pipeline._running = True
        pipeline._sdnl_worker()

        self.assertEqual(len(recorded_max_conf), 1)
        self.assertEqual(recorded_max_conf[0], 0.0)
        self.assertTrue(pipeline._alert_queue.empty())
        self.assertEqual(pipeline._attack_count, 0)

        # Contrast check: an unsuppressed ATTACK does raise max_confidence and publishes
        mock_sdnl.infer.return_value = ("ATTACK", 0.85)
        pipeline._roi_queue.put(bundle)
        pipeline._running = True
        pipeline._sdnl_worker()

        self.assertEqual(len(recorded_max_conf), 2)
        self.assertAlmostEqual(recorded_max_conf[1], 0.85, places=2)
        self.assertFalse(pipeline._alert_queue.empty())
        self.assertEqual(pipeline._attack_count, 1)


if __name__ == "__main__":
    unittest.main()
