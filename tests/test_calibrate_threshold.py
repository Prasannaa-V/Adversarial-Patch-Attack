import os
import tempfile
import unittest
import numpy as np

from training.calibrate_threshold import frame_maxes


class TestCalibrateThreshold(unittest.TestCase):
    def test_parses_new_nine_column_format(self):
        csv_content = (
            "timestamp,frame_id,bbox,tile,region,conf,label,published,max_region\n"
            "1791952276.155,frame_a,0,0,whole,0.0150,CLEAN,0,none\n"
            "1791952276.155,frame_a,0,1,top_half,0.0070,CLEAN,0,none\n"
            "1791952276.155,frame_a,0,2,square_2,0.8500,ATTACK,1,bbox0:person:square_2\n"
            "1791952277.200,frame_b,0,0,whole,0.0210,CLEAN,0,none\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
            f.write(csv_content)
            tmp_path = f.name

        try:
            maxes = frame_maxes([tmp_path])
            self.assertEqual(len(maxes), 2)
            self.assertAlmostEqual(maxes[0], 0.0210, places=4)
            self.assertAlmostEqual(maxes[1], 0.8500, places=4)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_parses_legacy_six_column_format(self):
        csv_content = (
            "timestamp,frame_id,bbox,tile,conf,published\n"
            "1791952276.155,frame_1,0,0,0.1200,0\n"
            "1791952276.155,frame_1,0,1,0.6400,1\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
            f.write(csv_content)
            tmp_path = f.name

        try:
            maxes = frame_maxes([tmp_path])
            self.assertEqual(len(maxes), 1)
            self.assertAlmostEqual(maxes[0], 0.6400, places=4)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
