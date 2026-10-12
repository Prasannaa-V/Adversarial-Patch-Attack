"""ROI tiling helper + composite attack-set contract (no training needed)."""
import glob
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.pipeline import DualModelPipeline
from inference.sdnl import person_regions


class TestPersonRegions(unittest.TestCase):
    def test_non_person_yields_whole_only(self):
        roi = np.zeros((128, 128, 3), dtype=np.uint8)
        out = list(person_regions(roi, "laptop"))
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], roi)

    def test_person_yields_whole_plus_halves(self):
        roi = np.random.default_rng(0).integers(0, 255, (128, 128, 3)).astype(np.uint8)
        whole, top, bot = list(person_regions(roi, "person"))
        self.assertEqual(top.shape, (64, 128, 3))
        self.assertEqual(bot.shape, (64, 128, 3))
        self.assertTrue(np.array_equal(np.concatenate([top, bot], axis=0), roi))

    def test_tiny_roi_skips_halves(self):
        roi = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(len(list(person_regions(roi, "person"))), 1)


class TestSplitTiles(unittest.TestCase):
    def test_tiles_1_is_identity(self):
        roi = np.zeros((160, 320, 3), dtype=np.uint8)
        out = DualModelPipeline.split_tiles(roi, 1)
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], roi)

    def test_tiles_2_covers_full_roi(self):
        roi = np.random.default_rng(0).integers(0, 255, (160, 320, 3)).astype(np.uint8)
        out = DualModelPipeline.split_tiles(roi, 2)
        self.assertEqual(len(out), 4)
        for t in out:
            self.assertEqual(t.shape, (80, 160, 3))
        # union of tiles reconstructs the ROI
        top = np.concatenate(out[:2], axis=1)
        bot = np.concatenate(out[2:], axis=1)
        self.assertTrue(np.array_equal(np.concatenate([top, bot], axis=0), roi))


class TestCompositeSet(unittest.TestCase):
    PREP_ATTACK = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed", "stress_attack")

    def test_expected_count(self):
        files = glob.glob(os.path.join(self.PREP_ATTACK, "comp_*.npy"))
        self.assertEqual(len(files), 600)

    def test_tensor_shapes_and_range(self):
        sample = sorted(glob.glob(os.path.join(self.PREP_ATTACK, "comp_*.npy")))[:10]
        self.assertTrue(sample)
        for p in sample:
            t = np.load(p)
            self.assertEqual(t.shape, (128, 128, 4), msg=p)
            self.assertTrue(np.all(t >= 0) and np.all(t <= 1), msg=p)


if __name__ == "__main__":
    unittest.main()
