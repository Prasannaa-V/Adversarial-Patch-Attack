"""QR structured-code suppression (mocked detector; no extra deps)."""
import os
import sys
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
