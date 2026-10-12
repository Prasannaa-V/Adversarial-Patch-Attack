#!/usr/bin/env python3
"""Overlay a patch onto a photo for honest demos (feathered, no edge cheating).

A pasted patch only flags if it is LARGE in the scored ROI: measured
half-box ~0.55 ATTACK, quarter-box ~0.25 miss. Default 0.40 of min dimension.
seamlessClone feathering avoids sharp rectangle edges the model could key on.

Usage:
  venv/bin/python overlay_demo.py --photo room.jpg --patch data/crops/attack/apricot_test_600877_0.jpg
  venv/bin/python overlay_demo.py --photo room.jpg --scale 0.5 --pos 0.5 0.4 --out demo_overlay.png
  venv/bin/python test_custom_images.py demo_overlay.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Feathered patch overlay for demos")
    ap.add_argument("--photo", required=True)
    ap.add_argument("--patch", default=os.path.join(ROOT_DIR, "data", "crops", "attack",
                                                    "apricot_test_600042_0.jpg"),
                    help="patch crop (default: strongest-scoring tight crop, 1.000)")
    ap.add_argument("--scale", type=float, default=0.40, help="patch side as fraction of photo min dimension")
    ap.add_argument("--pos", type=float, nargs=2, default=(0.5, 0.5), metavar=("X", "Y"),
                    help="patch center as fractions of width/height")
    ap.add_argument("--out", default="demo_overlay.png")
    args = ap.parse_args()

    photo = cv2.imread(args.photo)
    patch = cv2.imread(args.patch)
    if photo is None:
        raise SystemExit(f"cannot read photo: {args.photo}")
    if patch is None:
        raise SystemExit(f"cannot read patch: {args.patch}")
    h, w = photo.shape[:2]
    side = max(32, int(min(h, w) * args.scale))
    patch = cv2.resize(patch, (side, side), interpolation=cv2.INTER_LINEAR)

    cx, cy = int(w * args.pos[0]), int(h * args.pos[1])
    # Manual feathered alpha blend (NOT seamlessClone: Poisson cloning adapts
    # interior colors and destroys the adversarial texture being demoed).
    r = max(3, side // 12)
    alpha = np.zeros((side, side), np.float32)
    alpha[r:-r, r:-r] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), r / 2.0)[..., None]
    cx = min(max(cx, side // 2), w - side // 2)
    cy = min(max(cy, side // 2), h - side // 2)
    x0 = min(max(cx - side // 2, 0), w - side)
    y0 = min(max(cy - side // 2, 0), h - side)
    region = photo[y0:y0 + side, x0:x0 + side]  # exactly side×side by construction
    photo[y0:y0 + side, x0:x0 + side] = (
        patch.astype(np.float32) * alpha + region * (1.0 - alpha)).astype(np.uint8)
    out = photo
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out} ({w}x{h}, patch {side}px = {side / min(h, w):.0%} of min dim)")


if __name__ == "__main__":
    main()
