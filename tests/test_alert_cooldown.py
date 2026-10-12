"""Alert cooldown: one continuous patch presentation = one incident."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.pipeline import DualModelPipeline, PipelineConfig


class TestAlertCooldown(unittest.TestCase):
    def make_pipeline(self, cooldown=15.0, threshold=0.70):
        cfg = PipelineConfig(sdnl_threshold=threshold, alert_cooldown_s=cooldown)
        return DualModelPipeline(cfg)

    def test_first_attack_publishes(self):
        p = self.make_pipeline()
        self.assertTrue(p.should_publish_attack(0.95, 1000.0))
        self.assertEqual(p._attack_count, 1)

    def test_same_patch_within_cooldown_suppressed(self):
        p = self.make_pipeline(cooldown=15.0)
        self.assertTrue(p.should_publish_attack(0.95, 1000.0))
        # Holding the patch in view: next 14 s of frames produce nothing
        for t in (1001.0, 1005.0, 1010.0, 1014.9):
            self.assertFalse(p.should_publish_attack(0.97, t))
        self.assertEqual(p._attack_count, 1)

    def test_new_presentation_after_cooldown_publishes(self):
        p = self.make_pipeline(cooldown=15.0)
        self.assertTrue(p.should_publish_attack(0.95, 1000.0))
        self.assertTrue(p.should_publish_attack(0.95, 1015.0))
        self.assertEqual(p._attack_count, 2)

    def test_below_threshold_never_publishes(self):
        p = self.make_pipeline()
        self.assertFalse(p.should_publish_attack(0.10, 1000.0))
        self.assertFalse(p.should_publish_attack(0.69, 2000.0))
        self.assertEqual(p._attack_count, 0)

    def test_cooldown_configurable(self):
        p = self.make_pipeline(cooldown=60.0)
        self.assertTrue(p.should_publish_attack(0.95, 1000.0))
        self.assertFalse(p.should_publish_attack(0.95, 1059.0))
        self.assertTrue(p.should_publish_attack(0.95, 1060.0))


if __name__ == "__main__":
    unittest.main()
