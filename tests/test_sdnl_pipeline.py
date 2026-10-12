"""SDNL guard rails + pipeline demo/threshold defaults."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from sdnl import SDNLClassifier


class TestSdnlPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = SDNLClassifier(threshold=0.70)

    def test_default_threshold_is_070(self):
        self.assertEqual(SDNLClassifier.__init__.__defaults__[1], 0.70)

    def test_zeros_roi_is_clean_not_crash(self):
        label, conf = self.clf.infer(np.zeros((128, 128, 3), dtype=np.uint8))
        self.assertEqual(label, "CLEAN")
        self.assertEqual(conf, 0.0)

    def test_none_roi_is_clean_not_crash(self):
        label, conf = self.clf.infer(None)
        self.assertEqual(label, "CLEAN")

    def test_tiny_roi_is_clean_not_crash(self):
        label, _ = self.clf.infer(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertEqual(label, "CLEAN")

    def test_pipeline_demo_defaults(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from inference.pipeline import PipelineConfig
        cfg = PipelineConfig()
        self.assertEqual(cfg.sdnl_threshold, 0.70)
        self.assertEqual(cfg.demo_attack_every, 0)
        cfg.demo_attack_every = 10
        self.assertEqual(cfg.demo_confidence, 0.95)
        self.assertEqual(cfg.alert_cooldown_s, 15.0)


if __name__ == "__main__":
    unittest.main()
