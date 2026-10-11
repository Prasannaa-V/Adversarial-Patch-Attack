# 🚀 SESSION STARTER — InfraEdge-Shield

> **Read this first.** It reflects the ACTUAL as-built system (verified against the code on
> 2026-08-18), not the original design docs. Several older docs describe a superseded
> RPi + OAK-D design — see "Superseded docs" below.

**Project:** Real-Time Detection & Mitigation of Physical Adversarial Patch Attacks in
Edge-Based Smart Infrastructure.
**Goal:** Working system + top-tier paper (IEEE Trans. IoT / Computers & Security) +
India provisional patent (patent-FIRST).
**Status:** ~50% complete.

---

## Actual architecture (as-built)

A single workstation/laptop runs the full detection pipeline; an ESP32 is the fail-secure
gate actuator. No Raspberry Pi, no OAK-D, no depth sensor.

- **Camera:** generic USB webcam or RTSP/HTTP IP camera (monocular, no depth).
- **Detector:** YOLOv8n (Ultralytics) via OpenCV — `inference/detector.py`.
- **SDNL patch classifier:** 4-channel input **RGB + DCT high-frequency energy**
  (NO Laplacian, NO depth) → MobileNetV3-Small (<500K params), ONNX Runtime backend —
  `inference/sdnl.py`. 128×128 ROI, decision threshold 0.70 (config default; 0.703 F1-optimal operating point).
- **Pipeline:** 3 threads (frame reader → SDNL worker → alert publisher) with bounded
  queues — `inference/pipeline.py`. Classifies each ROI independently per frame
  (no temporal tracking yet).
- **Actuator:** ESP32 relay, fail-secure. Boots LOCKED, 15 s heartbeat watchdog, 30 s
  auto-release, manual override button — `esp32/firmware/src/main.cpp`.
- **Messaging:** Mosquitto MQTT. Topics `smartgate/{attack_alert, attack_ack, override,
  heartbeat, status}`. QoS1 for safety-critical, QoS0 for heartbeat.
- **Backend:** Node/Express + Socket.IO + MQTT bridge + SQLite — `dashboard/backend/`.
  Serves frames statically; POST /api/events ingests alerts and emits `attack_alert`.
- **Frontend:** React 19 + Vite, single `App.jsx` — `dashboard/frontend/`. Deps: react,
  socket.io-client, lucide-react only (no chart lib / router / axios).

### Data flow
camera → YOLOv8n (ROIs) → SDNL (RGB+DCT) → if p ≥ 0.70: MQTT `attack_alert` (QoS1) +
HTTP POST /api/events → ESP32 locks relay + ACK; backend stores event + Socket.IO pushes
to dashboard. PC sends a heartbeat every 5 s; if it stops, the ESP32 watchdog auto-locks
within 15 s (fail-secure).

---

## Three known critical issues (resolve before paper/patent)

1. **[BLOCKER] Dataset is not paper-valid.** `data/apde/` holds APRICOT-derived crops +
   synthetic fabricated images, NOT the real APDE dataset. Clean vs attack are adjacent
   crops of the *same* images, so the reported ~98.5% acc / ~1.05% FPR are NOT trustworthy.
   Real APDE + an independent clean FPR set (COCO val2017 / INRIA) are required, then
   retrain + re-benchmark. **Cannot be fixed by editing code** — needs data + a training run.
2. **[SECURITY] Edge actuation is spoofable.** MQTT is anonymous + no TLS and POST /api/gate
   is unauthenticated — anyone on the LAN can forge `override:unlock` or spoof heartbeats to
   defeat the watchdog. This is also the seed for patent novelty **N-C** (Zero-Trust Edge
   Attestation).
3. **No prior version control** — fixed 2026-08-18 (`git init` + `.gitignore`; baseline committed).

---

## Superseded docs (historical only — do NOT implement from them)
`docs/architecture_plan.md`, `docs/build_plan.md`, `docs/plans.md`, `docs/setup.md`,
`docs/feasibility.md` describe the abandoned RPi 4B + OAK-D stereo-depth design
(5-channel RGB+DCT+Laplacian, TFLite INT8, CRA/TypeScript, `edge/` / `embedded/` dirs).
Each now carries a SUPERSEDED banner. `docs/paper_sections.md` remains authoritative for the paper.

---

## Next session
Open `docs/progress.md` and work the top unchecked **Phase A** item. Phase A (integrity +
fixes) must finish before Phase B novelties, because the paper's numbers depend on a valid
dataset.
