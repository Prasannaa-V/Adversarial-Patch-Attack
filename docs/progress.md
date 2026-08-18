# ✅ PROGRESS — InfraEdge-Shield

_Last verified against code: 2026-08-18. Companion to `docs/session_starter.md`._

Legend: `[x]` done · `[ ]` todo · `[~]` partial · `[!]` blocked / needs external action

---

## DONE (verified in code)
- [x] YOLOv8n detector: webcam / RTSP / image / folder / dummy modes + graceful fallback — `inference/detector.py`
- [x] SDNL 4-channel (RGB + DCT) MobileNetV3-Small classifier, ONNX/PyTorch/heuristic backends — `inference/sdnl.py`
- [x] 3-thread pipeline (reader → SDNL → publisher), bounded queues, MQTT + HTTP POST + heartbeat — `inference/pipeline.py`
- [x] MQTT client (QoS1 alert, QoS0 heartbeat) — `inference/mqtt_client.py`
- [x] ESP32 fail-secure firmware: boot-locked, 15 s watchdog, 30 s auto-release, override button — `esp32/firmware/src/main.cpp`
- [x] Node backend: Express + Socket.IO + MQTT bridge + SQLite; POST /api/events emits `attack_alert` — `dashboard/backend/`
- [x] React 19 + Vite dashboard (single `App.jsx`): live alerts, ack, gate/health status — `dashboard/frontend/`
- [x] Training: FocalLoss, confidence-threshold tuning (target FPR 5%), ONNX export — `training/train_sdnl.py`
- [x] APRICOT parser + 4-ch preprocessing + latency benchmark harness — `training/`

## PHASE A — Integrity & fixes  ← DO THIS FIRST
- [!] **Real dataset.** Obtain real APDE (~94K imgs / 94 patch types); build an independent
      clean FPR set from COCO val2017 or INRIA; move the current synthetic/APRICOT crops into
      a clearly-labeled sanity-check set; retrain and re-benchmark; replace all reported
      metrics. _Blocked: needs a data download + retrain — cannot be done by editing code._
- [x] `git init` + `.gitignore`; baseline snapshot committed (2026-08-18)
- [x] ESP32 buzzer made non-blocking — removed `delay(2000)` from the MQTT-callback path;
      tone now started by `startBuzzer()` and stopped from `loop()` via `checkBuzzer()`, so
      the callback returns immediately and ACK latency stays well under 200 ms
- [x] `config/config.yaml` → `sdnl_model` now points to `models/sdnl/sdnl_v1.onnx`
      (was a non-existent `.tflite`; only worked via the hardcoded ONNX fallback)
- [x] `mqtt_bridge.js` now emits `attack_alert` on the DB-backup path, so the dashboard
      updates even if the pipeline's HTTP POST to /api/events fails
- [x] `detector.py` — replaced deprecated `datetime.utcnow()` (Py 3.14) with timezone-aware
      `datetime.now(timezone.utc)`
- [ ] Build a real clean FPR evaluation split and report FPR on it separately
- [ ] Re-run `training/benchmark_sdnl.py` on real ROIs (not random tensors); record edge latency

## PHASE B — Novelties (paper + patent)
- [ ] **N-A: Adaptive Spatio-Spectral Attention Fusion (SSAF)** — two-branch (RGB + DCT)
      MobileNetV3 with a learned attention gate, replacing naive 4-channel stacking.
      `inference/sdnl.py` + `training/train_sdnl.py`.
- [ ] **N-B: Temporal-Spatial Consistency Gating (TSCG)** — lightweight bbox tracking +
      K-of-N persistence before firing an alert; suppresses single-frame false positives.
      `inference/pipeline.py`.
- [ ] **N-C: Zero-Trust Edge Attestation** — HMAC-SHA256 + monotonic nonce on MQTT safety
      topics + JWT on POST /api/gate. Highest patent value; helps clear Indian Patent Act
      §3(k). `inference/mqtt_client.py`, `mqtt_bridge.js`, `main.cpp`, `gate.js`.

## PHASE C — Live video
- [ ] Backend `/stream` MJPEG endpoint fed by the pipeline
- [ ] Real live view in `App.jsx` (replace the placeholder animation) with alert overlay

## PHASE D — Benchmarking & paper
- [ ] Baselines (e.g., SentiNet / LGS / naive JPEG defense) vs SSAF+TSCG on valid data
- [ ] Latency/throughput + FPR/recall tables; ablations (3-ch vs 4-ch; with/without TSCG)
- [ ] Patent claims draft (N-C centered) + paper contributions section

---

## Blockers (2026-08-18)
| # | Blocker | Impact | Needs |
|---|---------|--------|-------|
| 1 | `data/apde` is APRICOT-derived + synthetic, not real APDE; clean/attack are same-image crops | All accuracy/FPR numbers are not paper-valid | Download real APDE + clean COCO/INRIA; retrain |
| 2 | Anonymous no-TLS MQTT + unauthenticated /api/gate | Actuator can be spoofed (forge unlock / fake heartbeat) | Implement N-C (HMAC/nonce + JWT) |

## Key numbers — ⚠️ measured on non-representative data; re-run before citing
| Metric | Value | Note |
|--------|-------|------|
| SDNL accuracy | ~98.5% | on same-image crops — NOT trustworthy |
| FPR | ~1.05% | no independent clean set — NOT trustworthy |
| SDNL params | <500K | MobileNetV3-Small, 4-channel |
| ROI size | 128×128 | |
| Decision threshold | 0.85 | tuned for ~5% target FPR |
| SDNL latency target | ≤15 ms | benchmarked on random tensors, not real ROIs |
