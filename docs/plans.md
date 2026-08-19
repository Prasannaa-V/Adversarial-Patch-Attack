# 📋 PROJECT PLANS & ROADMAP

## Smart Infrastructure Cyber-Physical Access Control System — Technical Roadmap

_Last updated: 2026-08-19. Aligned with current codebase, `docs/session_starter.md`, `docs/progress.md`, and `docs/paper_sections.md`._

---

## 1. Project Overview

**Problem:** Smart infrastructure (parking gates, secure entryways, access control points) relies heavily on edge-deployed computer vision models that are critically vulnerable to physical adversarial patch attacks — printed visual patterns that manipulate primary object detectors into misclassifying authorized subjects or inducing target blindness — with no cloud-independent, hardware-level failsafe defense.

**Proposed Solution:** A lightweight dual-model edge architecture:
- **Primary Detector:** YOLOv8n running locally via OpenCV (person/vehicle detection).
- **Secondary Layer (SDNL):** Custom sub-500K parameter 4-channel CNN (RGB + 2D DCT high-frequency spatial spectrum map `[h//2:, w//2:]`) for detecting adversarial patch noise signatures in < 2 ms.
- **Hardware-in-the-Loop Failsafe:** ESP32 microcontroller with Zero-Trust HMAC-SHA256 payload attestation → 5V relay → 12V electromagnetic gate lock (de-energized LOW = LOCKED fail-secure posture, < 60 ms response time).
- **Security Dashboard:** Full-featured React 19 + Node.js + Express + Socket.IO + SQLite application with live telemetry, incident gallery, JWT operator authentication (`/api/auth/login`), and manual override controls.

**Camera / Sensor:** Generic USB Webcam / RTSP IP Camera stream (no proprietary stereo-depth hardware dependency).

**Dataset:** Unpolluted independent dataset split (200 independent clean crops from COCO val2017 + 873 physical adversarial patch crops from APRICOT v1.0, zero image-level overlap).

---

## 2. Novelty Claims & Contributions

| # | Claim | Type | Status |
|---|---|---|---|
| N1 | Dual-concurrent model (primary YOLOv8n detector + sub-500K SDNL) running locally on edge host without cloud dependence | System | ✅ Implemented & Benchmarked |
| N2 | SDNL trained on 4-channel spatial-frequency tensors (RGB + 2D DCT spectrum map) for physical patch detection | Method | ✅ Implemented & Retrained |
| N3 | Hardware-in-the-loop: CV detection → Zero-Trust HMAC MQTT → ESP32 mbedTLS → de-energized fail-secure relay lock (< 60 ms) | System | ✅ Implemented & Attested |
| N4 | Edge security dashboard with live telemetry, JWT operator auth (`/api/auth/login`), SQLite deduplication, and manual override | System | ✅ Operational |
| N5 | Empirical performance validation on unpolluted independent clean COCO vs physical patch APRICOT test set (96.27% accuracy, 1.66 ms ONNX CPU latency) | Benchmark | ✅ Verified |

---

## 3. Milestone Timeline & Execution Status

| Milestone | Description | Deliverable | Status |
|---|---|---|---|
| **M0** | Planning + Lit Survey | `docs/` files, literature review table | ✅ COMPLETED |
| **M1** | Environment Setup | Python virtual environment, dependencies, dataset directories | ✅ COMPLETED |
| **M2** | Unpolluted Dataset Preparation | `prepare_clean_independent_dataset.py` (purged stale tensors; 200 COCO clean + 873 APRICOT attack) | ✅ COMPLETED |
| **M3** | SDNL Training & ONNX Export | Trained 4-channel MobileNetV3-Small (`train_sdnl.py`), exported ONNX graph (`sdnl_v1.onnx`) | ✅ COMPLETED |
| **M4** | Primary Detector Integration | YOLOv8n integration in `inference/detector.py` with CPU fallback | ✅ COMPLETED |
| **M5** | Multi-Threaded Pipeline | `inference/pipeline.py` (Reader → SDNL Worker → Alert Publisher) with thread locks & async POST | ✅ COMPLETED |
| **M6** | ESP32 Hardware Failsafe | `esp32/firmware/src/main.cpp` (PlatformIO, de-energized LOW = LOCKED, mbedTLS HMAC verification, 15s watchdog) | ✅ COMPLETED |
| **M7** | Dashboard Backend | Node.js + Express + Socket.IO + SQLite with `UNIQUE(frame_id)` deduplication & JWT auth | ✅ COMPLETED |
| **M8** | Dashboard Frontend | React 19 + Vite dashboard (`dashboard/frontend/src/App.jsx`) with live telemetry & override controls | ✅ COMPLETED |
| **M9** | Latency Benchmarking | `training/benchmark_sdnl.py` on real ROI tensors (1.66 ms ONNX CPU latency) | ✅ COMPLETED |
| **M10** | Technical Documentation & Paper | `docs/paper_sections.md` updated with verified empirical metrics & fail-secure relay specs | ✅ COMPLETED |

---

## 4. Actual Project Directory Structure

```
/home/prasannaa/Projects/p1/
├── docs/                        # Technical documentation & planning
│   ├── paper_sections.md        # Authoritative capstone technical paper & specification
│   ├── session_starter.md      # Grounding summary & system state
│   ├── progress.md             # Verified task completion log & empirical metrics
│   ├── plans.md                # Project roadmap & milestone plan (THIS FILE)
│   └── diagram_sources.md      # PlantUML & Eraser.io architecture diagram sources
│
├── data/                        # Datasets & preprocessed tensors (ignored in git)
│   ├── apricot/                 # APRICOT physical patch dataset
│   ├── coco_clean/              # COCO val2017 independent clean images
│   ├── crops/                   # Extracted 128x128 crop images (clean/ & attack/)
│   ├── preprocessed/            # 4-channel float32 .npy tensors (clean/ & attack/)
│   └── frames/                  # Flagged incident snapshot images
│
├── models/                      # Trained model files (ignored in git)
│   └── sdnl/
│       ├── sdnl_v1.pt           # PyTorch checkpoint
│       └── sdnl_v1.onnx         # Primary ONNX CPU inference model
│
├── inference/                   # Python inference pipeline
│   ├── detector.py              # YOLOv8n primary detector module
│   ├── sdnl.py                  # 4-channel tensor builder & ONNX/PyTorch classifier
│   ├── pipeline.py              # 3-thread pipeline (reader, worker, publisher)
│   ├── mqtt_client.py           # Zero-Trust HMAC-SHA256 MQTT client wrapper
│   └── frame_store.py           # Flagged frame saver & 7-day rolling disk cleanup
│
├── esp32/                       # ESP32 failsafe firmware
│   └── firmware/
│       ├── src/main.cpp         # Arduino C++ firmware (mbedTLS HMAC, LOW=LOCKED fail-secure)
│       └── platformio.ini       # PlatformIO build configuration
│
├── dashboard/                   # Web security dashboard
│   ├── backend/                 # Node.js + Express + Socket.IO + SQLite
│   │   ├── server.js            # Express server entry point
│   │   ├── routes/              # REST API routes (auth.js, gate.js, events.js, stats.js)
│   │   ├── services/            # Services (db.js, mqtt_bridge.js with HMAC signing)
│   │   └── db/schema.sql        # SQLite schema with UNIQUE(frame_id)
│   └── frontend/                # React 19 + Vite dashboard
│       └── src/App.jsx          # React single-page dashboard application
│
├── training/                    # Model training & benchmark scripts
│   ├── train_sdnl.py            # SDNL training script (seed=42, CrossEntropy, ONNX export)
│   ├── parse_apricot.py         # APRICOT physical patch crop extractor
│   ├── download_real_clean_coco.py # COCO val2017 clean downloader & tensor builder
│   ├── prepare_clean_independent_dataset.py # Disk purge & unpolluted dataset builder
│   └── benchmark_sdnl.py        # Latency benchmark on real ROI tensors
│
├── config/                      # System configuration
│   └── config.yaml              # YAML config file (smartgate/* topics, 7-day retention)
│
└── .gitignore                   # Git ignore file (venv, models, datasets, images, db, logs)
```

---

## 5. Verified Success Metrics & Empirical Baseline

| Metric | Target | Verified Empirical Result | Status |
|---|---|---|---|
| SDNL Test Accuracy | ≥ 92% | **96.27%** on unpolluted test set | ✅ PASS |
| SDNL Test Precision | ≥ 90% | **100.00%** (0.00% FPR on clean COCO crops) | ✅ PASS |
| SDNL Test Recall | ≥ 90% | **95.42%** on physical APRICOT patch crops | ✅ PASS |
| SDNL F1-Score | ≥ 0.90 | **0.9766** | ✅ PASS |
| SDNL ROC-AUC | ≥ 0.95 | **0.9880** | ✅ PASS |
| SDNL ROI ONNX Latency (CPU) | ≤ 15 ms | **1.66 ms** per ROI crop (P95: 2.58 ms) | ✅ PASS |
| End-to-End Latency (Camera → ESP32 Relay) | ≤ 200 ms | **< 60 ms** hardware actuation | ✅ PASS |
| System Throughput | ≥ 15 FPS | **20+ FPS** multi-threaded pipeline | ✅ PASS |
| Hardware Security Posture | Fail-Secure | **De-energized LOW = LOCKED** on MCU reset | ✅ PASS |
| Messaging Security | Attested | **Zero-Trust HMAC-SHA256** (Python, JS, ESP32 mbedTLS) | ✅ PASS |
