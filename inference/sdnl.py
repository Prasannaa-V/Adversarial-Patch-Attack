"""
inference/sdnl.py
Secondary Diagnostic Neural Layer — 4-channel CNN classifier.
Inputs: RGB ROI (128x128) + DCT high-frequency map.
Output: ('CLEAN' or 'ATTACK', confidence: float)

Supports:
- ONNX Runtime (.onnx) — primary high-performance CPU/edge backend
- PyTorch (.pt / .pth) — direct PyTorch model execution
- TFLite (.tflite) — mobile/microcontroller edge runtime
- DCT Energy Heuristic — offline fallback if no model file is found
"""

import cv2
import numpy as np
import logging
import time
import os
from typing import Tuple

logger = logging.getLogger(__name__)


def build_4ch_tensor(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Build 4-channel input tensor from a 128x128 BGR ROI.
      Ch 1-3: Normalized RGB (float32, 0-1)
      Ch 4:   2D DCT high-frequency spatial spectrum map (float32, 0-1)

    Args:
        roi_bgr: np.ndarray, shape (128, 128, 3), uint8 BGR image crop

    Returns:
        tensor: np.ndarray, shape (128, 128, 4), float32
    """
    roi = cv2.resize(roi_bgr, (128, 128))
    # Explicit BGR to RGB conversion for model compatibility
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    rgb = roi_rgb.astype(np.float32) / 255.0  # (128, 128, 3)

    # Grayscale for 2D Discrete Cosine Transform (DCT) frequency analysis
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2D DCT — extract high-frequency 2D spatial spectrum quadrant (bottom-right: high horizontal & vertical frequencies)
    dct = cv2.dct(gray)
    high_freq = np.zeros_like(dct)
    h, w = dct.shape
    high_freq[h // 2:, w // 2:] = dct[h // 2:, w // 2:]

    # Normalize DCT channel to [0, 1]
    dct_abs = np.abs(high_freq)
    dct_max = dct_abs.max()
    if dct_max > 0:
        dct_norm = dct_abs / dct_max
    else:
        dct_norm = dct_abs  # all zeros

    # Stack into 4-channel tensor: (H, W, 4)
    tensor = np.concatenate([rgb, dct_norm[:, :, np.newaxis]], axis=2)
    return tensor.astype(np.float32)


def qr_code_present(roi_bgr: np.ndarray) -> bool:
    """Detect structured QR codes (badges, posters, gate passes).

    QR patterns are dense high-frequency black-and-white grids, so they
    legitimately excite the DCT channel. Suppress them here instead of
    retraining: a decodable/locatable QR is a structured code, not a patch.
    Returns False on any detector failure (fail-open toward SDNL verdict).
    """
    try:
        if roi_bgr is None or not isinstance(roi_bgr, np.ndarray) or roi_bgr.size == 0:
            return False
        h, w = roi_bgr.shape[:2]
        if h < 32 or w < 32:
            return False
        probe = roi_bgr if max(h, w) <= 256 else cv2.resize(
            roi_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(probe)
        # Require valid decoded QR payload to prevent false suppression of textured adversarial patches
        return bool(data and len(data.strip()) > 0)
    except Exception:
        return False


def dct_energy_score(roi_rgb: np.ndarray) -> float:
    """
    Lightweight heuristic: normalized DCT high-frequency energy.
    Higher score = more high-frequency content = more likely adversarial.
    Used as fallback when deep learning models are unavailable.
    """
    roi = cv2.resize(roi_rgb, (128, 128))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dct = cv2.dct(gray)
    h, w = dct.shape
    total_energy = np.sum(dct ** 2) + 1e-9
    hf_energy = np.sum(dct[h // 2:, w // 2:] ** 2)
    ratio = hf_energy / total_energy
    # Normalize: typical clean images ~0.05, adversarial ~0.20+
    score = float(np.clip((ratio - 0.05) / 0.20, 0.0, 1.0))
    return score


class SDNLClassifier:
    """
    4-channel SDNL adversarial patch classifier.
    Supports ONNX, PyTorch, TFLite, and DCT Energy Heuristic fallback.
    """

    def __init__(self, model_path: str = "models/sdnl/sdnl_v1.onnx", threshold: float = 0.30,
                 qr_suppress: bool = True):
        self.threshold = threshold
        self.qr_suppress = qr_suppress
        self.model_path = model_path
        self._latency_ms = 0.0
        self._backend = "heuristic"
        self._ort_session = None
        self._torch_model = None
        self._tflite_interpreter = None

        self._load_model()

    def _load_model(self):
        # 1. Try ONNX model if specified or available in default locations
        onnx_candidates = [self.model_path, "models/sdnl/sdnl_v1.onnx"]
        for path in onnx_candidates:
            if path and os.path.exists(path) and path.endswith(".onnx"):
                try:
                    import onnxruntime as ort
                    self._ort_session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                    self._input_name = self._ort_session.get_inputs()[0].name
                    self._backend = "onnx"
                    logger.info(f"Loaded SDNL ONNX model from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize ONNX model from {path}: {e}")

        # 2. Try PyTorch model (.pt / .pth)
        pt_candidates = [self.model_path, "models/sdnl/sdnl_v1.pt", "models/sdnl/sdnl_best.pth"]
        for path in pt_candidates:
            if path and os.path.exists(path) and (path.endswith(".pt") or path.endswith(".pth")):
                try:
                    import torch
                    import sys
                    sys.path.append(os.path.join(os.path.dirname(__file__), '../training'))
                    from train_sdnl import SDNLModel

                    model = SDNLModel(num_classes=2, in_channels=4)
                    model.load_state_dict(torch.load(path, map_location="cpu"))
                    model.eval()
                    self._torch_model = model
                    self._backend = "pytorch"
                    logger.info(f"Loaded SDNL PyTorch model from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load PyTorch model from {path}: {e}")

        # 3. Try TFLite model (.tflite)
        tflite_candidates = [self.model_path, "models/sdnl/sdnl_v1.tflite"]
        for path in tflite_candidates:
            if path and os.path.exists(path) and path.endswith(".tflite"):
                try:
                    try:
                        import tflite_runtime.interpreter as tflite
                    except ImportError:
                        import tensorflow.lite as tflite

                    self._tflite_interpreter = tflite.Interpreter(model_path=path)
                    self._tflite_interpreter.allocate_tensors()
                    self._input_details = self._tflite_interpreter.get_input_details()
                    self._output_details = self._tflite_interpreter.get_output_details()
                    self._backend = "tflite"
                    logger.info(f"Loaded SDNL TFLite model from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load TFLite model from {path}: {e}")

        logger.warning("No trained deep learning checkpoint found. Falling back to DCT HEURISTIC mode.")
        self._backend = "heuristic"

    def infer(self, roi_rgb: np.ndarray) -> Tuple[str, float]:
        """
        Run inference on one ROI.

        Returns:
            label: 'CLEAN' or 'ATTACK'
            confidence: float in [0, 1]
        """
        t0 = time.perf_counter()

        # Guard: invalid / empty ROI (e.g. degenerate BBox) -> CLEAN, never crash
        if roi_rgb is None or not isinstance(roi_rgb, np.ndarray) or roi_rgb.size == 0:
            self._latency_ms = (time.perf_counter() - t0) * 1000.0
            return "CLEAN", 0.0
        if roi_rgb.shape[0] < 8 or roi_rgb.shape[1] < 8:
            self._latency_ms = (time.perf_counter() - t0) * 1000.0
            return "CLEAN", 0.0
        if np.count_nonzero(roi_rgb) == 0:
            self._latency_ms = (time.perf_counter() - t0) * 1000.0
            return "CLEAN", 0.0

        if self._backend == "onnx":
            tensor = build_4ch_tensor(roi_rgb)            # (128, 128, 4)
            # ONNX expects shape (N, C, H, W) -> (1, 4, 128, 128)
            tensor_ch_first = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
            outputs = self._ort_session.run(None, {self._input_name: tensor_ch_first})
            logits = outputs[0][0]
            probs = self._softmax(logits)
            attack_conf = float(probs[1])

        elif self._backend == "pytorch":
            import torch
            tensor = build_4ch_tensor(roi_rgb)
            tensor_ch_first = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
            torch_tensor = torch.tensor(tensor_ch_first, dtype=torch.float32)
            with torch.no_grad():
                logits = self._torch_model(torch_tensor).numpy()[0]
            probs = self._softmax(logits)
            attack_conf = float(probs[1])

        elif self._backend == "tflite":
            tensor = build_4ch_tensor(roi_rgb)[np.newaxis, ...] # (1, 128, 128, 4)
            self._tflite_interpreter.set_tensor(self._input_details[0]['index'], tensor)
            self._tflite_interpreter.invoke()
            output = self._tflite_interpreter.get_tensor(self._output_details[0]['index'])
            logits = output[0]
            probs = self._softmax(logits)
            attack_conf = float(probs[1])

        else: # Heuristic mode
            attack_conf = dct_energy_score(roi_rgb)

        self._latency_ms = (time.perf_counter() - t0) * 1000.0
        if attack_conf < self.threshold:
            return "CLEAN", attack_conf

        # Structured QR codes (badges/passes) excite the DCT channel but are
        # legitimate: suppress only on the ATTACK path so clean ROIs pay no cost.
        if self.qr_suppress and qr_code_present(roi_rgb):
            logger.info(f"QR structured-code suppress: SDNL={attack_conf:.3f} -> CLEAN")
            return "CLEAN", attack_conf
        return "ATTACK", attack_conf

    def get_latency_ms(self) -> float:
        return self._latency_ms

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        return e / e.sum()
