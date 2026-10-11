"""FrameStore retention + save/list round-trip (temp dir, no disk pollution)."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from frame_store import FrameStore


class TestFrameStore(unittest.TestCase):
    def test_cleanup_accepts_max_days_arg(self):
        with tempfile.TemporaryDirectory() as d:
            s = FrameStore(frames_dir=d)
            self.assertEqual(s.cleanup_old_frames(max_days=7), 0)
            self.assertEqual(s.cleanup_old_frames(), 0)

    def test_default_retention_is_7_days(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(FrameStore(frames_dir=d).max_days, 7)

    def test_save_and_list(self):
        with tempfile.TemporaryDirectory() as d:
            s = FrameStore(frames_dir=d)
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[10:20, 10:20] = 255
            path = s.save_frame(img, "unittest-frame")
            self.assertTrue(path and os.path.exists(path))
            self.assertGreaterEqual(len(s.list_frames()), 1)

    def test_old_frames_purged(self):
        with tempfile.TemporaryDirectory() as d:
            s = FrameStore(frames_dir=d, max_days=7)
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            path = s.save_frame(img, "old-frame")
            old = time.time() - 8 * 86400
            os.utime(path, (old, old))
            self.assertEqual(s.cleanup_old_frames(), 1)
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
