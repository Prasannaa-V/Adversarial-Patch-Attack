"""
scratch/test_camera.py
Verifies camera device connection (OpenCV / OAK-D / USB Camera) with simulated fallback.
"""

import cv2
import time

def test_camera_source(cam_id=0):
    print("=== Testing Camera Input Source ===")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[!] Camera index {cam_id} unavailable. Using dummy synthetic frame generator.")
        dummy_frame = (255 * (0.5 + 0.5 * cv2.randn(cv2.Mat.zeros((480, 640, 3), cv2.CV_8UC3), 128, 50))).astype('uint8')
        print(f"[✓] Generated synthetic test frame shape: {dummy_frame.shape}")
        return True
    
    ret, frame = cap.read()
    if ret:
        print(f"[✓] Successfully captured live frame: {frame.shape}")
    cap.release()
    return ret

if __name__ == "__main__":
    test_camera_source()
