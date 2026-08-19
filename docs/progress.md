# ✅ PROGRESS — Smart Physical Access Security System

_Last verified against code and disk: 2026-08-19. Companion to `docs/session_starter.md`._

Legend: `[x]` done · `[ ]` todo · `[~]` partial · `[!]` blocked / needs external action

---

## DONE (verified in code & disk)
- [x] YOLOv8n detector: webcam / RTSP / image / folder / dummy modes + graceful fallback — `inference/detector.py`
- [x] SDNL 4-channel (RGB + 2D DCT) MobileNetV3-Small classifier, ONNX/PyTorch/heuristic backends — `inference/sdnl.py`
- [x] 3-thread pipeline (reader → SDNL → publisher), bounded queues, thread-safe metrics lock, async alert posting, MQTT + HTTP POST + heartbeat — `inference/pipeline.py`
- [x] MQTT client with Zero-Trust HMAC-SHA256 Payload Attestation — `inference/mqtt_client.py`
- [x] ESP32 fail-secure firmware: de-energized LOW = LOCKED, 15 s watchdog, 30 s auto-release, non-blocking buzzer, GPIO33 pullup button — `esp32/firmware/src/main.cpp`
- [x] Node backend: Express + Socket.IO + JWT operator auth (`/api/auth/login`) + MQTT bridge + SQLite with `UNIQUE(frame_id)` deduplication + HMAC payload verification — `dashboard/backend/`
- [x] React 19 + Vite dashboard: live alerts, ack, gate/health status — `dashboard/frontend/`
- [x] Complete Disk Purge & Independent Clean Dataset Pipeline: `training/prepare_clean_independent_dataset.py` (Purged all 1,423 legacy background crops; independent COCO val2017 clean + APRICOT attack, zero same-image leakage)
- [x] Retrained SDNL model on purged unpolluted independent clean COCO vs physical APRICOT attack crops: 95.42% Recall, 100.00% Precision (0.00% FPR), 96.27% Accuracy, 0.9766 F1-Score, 0.9880 ROC-AUC — `training/train_sdnl.py`
- [x] Empirical Real ROI ONNX Benchmark: 1.66 ms mean latency on CPU (P95: 2.58 ms) — `training/benchmark_sdnl.py`

---

## Verified Empirical Benchmark Numbers (Purged Dataset 2026-08-19)
| Metric | Value | Status |
|--------|-------|--------|
| Test Recall (Detection Rate) | 95.42% | Verified on purged unpolluted test set |
| Test Precision | 100.00% | Verified (0.00% FPR on clean COCO crops) |
| Test Accuracy | 96.27% | Verified on purged unpolluted test set |
| Test F1-Score | 0.9766 | Verified on purged unpolluted test set |
| Test ROC-AUC | 0.9880 | Verified on purged unpolluted test set |
| False Positive Rate (FPR) | 0.00% | Verified at 0.70 threshold |
| Real ROI ONNX Latency | 1.66 ms | PASS (<= 15.0 ms per ROI) |
| Real ROI PyTorch Latency | 3.53 ms | PASS (<= 15.0 ms per ROI) |
| SDNL Parameters | < 500K | MobileNetV3-Small (4-channel) |
