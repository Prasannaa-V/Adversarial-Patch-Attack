"""Synthetic hard-negative generator: deterministic output contract."""
import glob
import os
import subprocess
import sys
import unittest

import numpy as np

PREP_CLEAN = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed", "clean")


class TestSynthNegatives(unittest.TestCase):
    def test_expected_family_counts(self):
        expected = {"text": 60, "icons": 60, "qr": 60, "barcode": 40,
                    "moire": 40, "halftone": 40, "fabric": 60}
        for fam, n in expected.items():
            files = glob.glob(os.path.join(PREP_CLEAN, f"synth_{fam}_*.npy"))
            self.assertEqual(len(files), n, msg=f"family {fam}")

    def test_tensor_shapes_and_range(self):
        sample = sorted(glob.glob(os.path.join(PREP_CLEAN, "synth_*.npy")))[:14]
        self.assertTrue(sample)
        for p in sample:
            t = np.load(p)
            self.assertEqual(t.shape, (128, 128, 4), msg=p)
            self.assertTrue(np.all(t >= 0) and np.all(t <= 1), msg=p)

    def test_deterministic_regeneration(self):
        before = {}
        for p in sorted(glob.glob(os.path.join(PREP_CLEAN, "synth_qr_00[0-2].npy"))):
            before[p] = np.load(p).sum()
        r = subprocess.run([sys.executable, "training/synth_hard_negatives.py"],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, msg=r.stderr[-500:])
        for p, s in before.items():
            self.assertAlmostEqual(float(np.load(p).sum()), float(s), places=4, msg=p)


if __name__ == "__main__":
    unittest.main()
