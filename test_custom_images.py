#!/usr/bin/env python3
"""
test_custom_images.py
Fast testing utility to evaluate custom downloaded images against the 4-channel SDNL ONNX model.
Mirrors the live pipeline: YOLO extracts ROIs first, then SDNL classifies each ROI.

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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "inference"))

from sdnl import SDNLClassifier, person_regions
from detector import YOLODetector  # exact pipeline proposal path (same boxes, same verdict)

THRESHOLD = 0.30   # matches pipeline default


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
    sdnl = SDNLClassifier(model_path=model_path, threshold=THRESHOLD)

    print(f"[*] Loading YOLOv8n for ROI extraction (pipeline detector)...")
    use_yolo = True
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

    print(f"\nEvaluating {len(image_paths)} image(s) [threshold={THRESHOLD:.0%}]:\n")
    header = f"{'Filename':<34} | {'YOLO Class':<17} | {'Result':<12} | {'SDNL Conf':>9} | {'Latency':>8}"
    print(header)
    print("-" * len(header))

    for p in image_paths:
        fname = os.path.basename(p)
        display_name = fname if len(fname) <= 32 else fname[:29] + "..."

        img = cv2.imread(p)
        if img is None:
            print(f"{display_name:<34} | {'ERROR':<17} | Could not read image")
            continue

        t0 = time.perf_counter()

        if use_yolo:
            # EXACT pipeline path: YOLODetector proposals (letterboxed) +
            # whole/halves scoring via person_regions. Same boxes, same verdict.
            from detector import YOLODetector
            det = YOLODetector(camera_source=p)
            bundle = det.get_frame()
            det.release()

            best_conf = 0.0
            best_label = "CLEAN"
            best_cls = ""
            for r in bundle.detections:
                roi = r.to_roi(bundle.rgb_frame)
                for region in person_regions(roi, r.label):
                    label, conf = sdnl.infer(region)
                    if conf > best_conf:
                        best_conf = conf
                        best_label = label
                        best_cls = (f"{r.label}({r.confidence:.0%})"
                                    if not r.label.startswith("camera_") else "fallback")
        else:
            best_label, best_conf = sdnl.infer(img)
            best_cls = "full_image"

        total_ms = (time.perf_counter() - t0) * 1000
        result_str = "ATTACK" if best_label == "ATTACK" else "CLEAN"
        print(f"{display_name:<34} | {best_cls:<17} | {result_str:<12} | {best_conf*100:>7.2f}%  | {total_ms:>6.1f} ms")

    print("-" * len(header))
    print(f"Done. (SDNL threshold: {THRESHOLD:.0%} | Pipeline: YOLO->ROI->SDNL)\n")


if __name__ == "__main__":
    main()
