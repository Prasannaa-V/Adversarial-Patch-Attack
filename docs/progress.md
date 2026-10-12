# ✅ PROGRESS — Smart Physical Access Security System

_Last verified against code and disk: 2026-10-12. Companion to `docs/session_starter.md`._

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
- [x] Retrained SDNL model on purged unpolluted independent clean COCO vs physical APRICOT attack crops — `training/train_sdnl.py`
- [x] Hard-negative retraining against texture FPs (phone screens, QR, text, icons, barcodes, moiré, halftone, fabric): `training/synth_hard_negatives.py` (360 synthetic CLEAN tensors) + retrained SDNL — held-out 99.53% acc / 100% prec / 99.23% recall @ 0.30, 0.00% FPR, synth-hard FPR 58% → 0.0%
- [x] Unified HMAC-SHA256 canonical form across Python/JS/ESP32 + signed ESP32 ACK/STATUS + verified heartbeats; HTTP ingestion requires HMAC, ack requires JWT — `inference/mqtt_client.py`, `dashboard/backend/services/mqtt_bridge.js`, `dashboard/backend/routes/events.js`, `esp32/firmware/src/main.cpp`
- [x] One-command demo mode (`./demo.sh`, `--demo-attack-every N`), exact `max-frames` stop, BGR annotation fix — `inference/pipeline.py`
- [x] Dashboard hourly attack chart (no new deps) + `GET /api/health` — `dashboard/frontend/src/App.jsx`, `dashboard/backend/server.js`
- [x] Smoke test suite (18 tests: HMAC vectors + PY↔JS cross-compat, FrameStore, SDNL guards, live auth) — `tests/`, run `venv/bin/python -m unittest discover -s tests -v`

---

## Verified Empirical Benchmark Numbers (Purged + hard-negative dataset, re-verified 2026-10-12 via `benchmark_sdnl.py --eval-only --threshold 0.30`)
| Metric | Value | Status |
|--------|-------|--------|
| Test Recall (Detection Rate) | 99.23% | Verified on held-out test set (130 attack / 84 clean) |
| Test Precision | 100.00% | Verified (0 FP in 214 test samples) |
| Test Accuracy | 99.53% | Verified on held-out test set |
| Test F1-Score | 0.9961 | Verified @ 0.30 operating threshold |
| Test AUC-PR / AUC-ROC | see `pr_curve.png` / `roc_curve.png` | Regenerated 2026-10-12 |
| False Positive Rate (FPR) | 0.00% | Verified @ 0.30 threshold (target ≤ 5%) |
| Hard-texture FPR (360 synth: text/icons/QR/barcode/moiré/halftone/fabric) | 0.0% @ 0.30 | Was ~58% before hard-negative retraining |
| Real ROI ONNX Latency | ≤ 15 ms target | PASS (edge-ready per-run stats in eval log) |
| SDNL Parameters | < 500K | MobileNetV3-Small (4-channel) |
