#!/usr/bin/env python3
"""
test_custom_images.py
Fast testing utility to evaluate custom downloaded images against the 4-channel SDNL ONNX model.

Usage:
  python test_custom_images.py path/to/image.jpg
  python test_custom_images.py path/to/folder/
  python test_custom_images.py img1.jpg img2.png img3.jpeg
"""

import sys
import os
import glob
import cv2
import time

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "inference"))

from sdnl import SDNLClassifier

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_custom_images.py <image_or_folder_path> ...")
        sys.exit(1)

    model_path = os.path.join(ROOT_DIR, "models/sdnl/sdnl_v1.onnx")
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"[*] Loading SDNL ONNX classifier ({model_path})...")
    sdnl = SDNLClassifier(model_path=model_path, threshold=0.70)

    # Gather images
    image_paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                image_paths.extend(glob.glob(os.path.join(arg, ext)))
                image_paths.extend(glob.glob(os.path.join(arg, "**", ext), recursive=True))
        elif os.path.isfile(arg):
            image_paths.append(arg)
        else:
            print(f"[!] Warning: Path not found: {arg}")

    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        print("[!] No valid image files found.")
        sys.exit(1)

    print(f"\nEvaluating {len(image_paths)} image(s):\n")
    print(f"{'Filename':<35} | {'Prediction':<12} | {'Confidence':<12} | {'Latency':<10}")
    print("-" * 78)

    for p in image_paths:
        fname = os.path.basename(p)
        if len(fname) > 33:
            fname = fname[:30] + "..."

        img = cv2.imread(p)
        if img is None:
            print(f"{fname:<35} | {'ERROR':<12} | {'Could not read image'}")
            continue

        t0 = time.perf_counter()
        label, conf = sdnl.infer(img)
        dt_ms = (time.perf_counter() - t0) * 1000

        pred_str = "🚨 ATTACK" if label == "ATTACK" else "✅ CLEAN"
        print(f"{fname:<35} | {pred_str:<12} | {conf*100:>6.2f}%     | {dt_ms:>6.2f} ms")

    print("-" * 78)
    print("Done. (Detection threshold: 70.0%)\n")

if __name__ == "__main__":
    main()
