"""Pick an operating threshold from live score captures.

Protocol: point the pipeline at the real camera and capture one CSV per
scenario with --score-log, e.g.
    pipeline.py --camera-url http://... --score-log clean.csv      # 1-2 min, no patch
    pipeline.py --camera-url http://... --score-log screen.csv     # phone/laptop screens
    pipeline.py --camera-url http://... --score-log patch.csv      # printed patch at 0.5-3 m
Then:
    python training/calibrate_threshold.py --clean clean.csv screen.csv --attack patch.csv

Uses each frame's max tile score (the alert decision value) and sweeps
thresholds, reporting FPR/FNR. Suggests the lowest threshold with FPR <= 5%,
plus the Youden-J optimum. numpy only.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def frame_maxes(paths):
    frames = defaultdict(float)
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    conf = float(row["conf"])
                except (KeyError, ValueError):
                    continue
                fid = row.get("frame_id", str(len(frames)))
                frames[fid] = max(frames.get(fid, 0.0), conf)
    return np.array(sorted(frames.values()))


def main():
    ap = argparse.ArgumentParser(description="Calibrate SDNL threshold from live captures")
    ap.add_argument("--clean", nargs="+", required=True, help="CSVs of non-attack scenes")
    ap.add_argument("--attack", nargs="+", required=True, help="CSVs with a real patch in view")
    ap.add_argument("--max-fpr", type=float, default=0.05)
    args = ap.parse_args()

    clean = frame_maxes(args.clean)
    attack = frame_maxes(args.attack)
    print(f"clean frames: {len(clean)} (max={clean.max():.3f} p95={np.percentile(clean, 95):.3f})")
    print(f"attack frames: {len(attack)} (min={attack.min():.3f} p5={np.percentile(attack, 5):.3f})")

    print(f"{'thresh':>7} {'FPR%':>7} {'FNR%':>7} {'YoudenJ':>8}")
    best_j, best_t, safe_t = -1.0, None, None
    for t in np.arange(0.05, 0.95, 0.05):
        fpr = float((clean >= t).mean()) if len(clean) else 0.0
        fnr = float((attack < t).mean()) if len(attack) else 0.0
        j = (1 - fnr) + (1 - fpr) - 1
        print(f"{t:7.2f} {fpr * 100:7.2f} {fnr * 100:7.2f} {j:8.3f}")
        if j > best_j:
            best_j, best_t = j, round(float(t), 2)
        if fpr <= args.max_fpr and safe_t is None:
            safe_t = round(float(t), 2)
    print(f"lowest threshold with FPR <= {args.max_fpr * 100:.0f}%: {safe_t}")
    print(f"Youden-J optimum: {best_t} (J={best_j:.3f})")


if __name__ == "__main__":
    main()
