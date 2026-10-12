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

from sdnl import SDNLClassifier
from detector import RELEVANT_CLASSES  # single source of truth, same as live pipeline

THRESHOLD = 0.30   # matches pipeline default
YOLO_CONF = 0.50   # matches live pipeline (looser values spray low-confidence boxes)
YOLO_IMGSZ = 416   # matches live pipeline


def get_yolo_rois(img, model, conf_cutoff=YOLO_CONF):
    """Run YOLOv8n and return list of (roi_crop, cls_name, yolo_conf, box)."""
    h, w = img.shape[:2]
    results = model(img, imgsz=YOLO_IMGSZ, conf=conf_cutoff, verbose=False)
    names = model.names
    rois = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        if cls_name not in RELEVANT_CLASSES:
            continue
        yconf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            continue
        roi = img[y1:y2, x1:x2]
        rois.append((roi, cls_name, yconf, (x1, y1, x2, y2)))

    # Fallback: if nothing relevant detected, use a center crop
    if not rois:
        cx, cy = w // 2, h // 2
        size = min(w, h) // 3
        roi = img[max(0, cy-size):cy+size, max(0, cx-size):cx+size]
        rois.append((roi, "center_fallback", 0.0, (cx-size, cy-size, cx+size, cy+size)))

    return rois


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

    print(f"[*] Loading YOLOv8n for ROI extraction...")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        use_yolo = True
    except Exception as e:
        print(f"[!] YOLO unavailable ({e}), will run SDNL on full image (less accurate).")
        yolo_model = None
        use_yolo = False

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
            rois = get_yolo_rois(img, yolo_model)

            # Run SDNL on each ROI; flag image if ANY roi is ATTACK
            best_conf = 0.0
            best_label = "CLEAN"
            best_cls = ""
            for roi, cls_name, yconf, box in rois:
                label, conf = sdnl.infer(roi)
                if conf > best_conf:
                    best_conf = conf
                    best_label = label
                    best_cls = (f"{cls_name}({yconf:.0%})"
                                if cls_name != "center_fallback" else "center_crop")
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
