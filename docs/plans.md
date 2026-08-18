# 📋 PROJECT PLANS
## InfraEdge-Shield — Full Roadmap & Milestone Plan

---

## 1. Project Overview

**Problem:** Smart infrastructure (parking gates, secure entryways) uses edge-deployed CV models that are critically vulnerable to physical adversarial patch attacks — printed visual patterns that cause misclassification or target blindness — with no cloud-independent, hardware-level failsafe defense.

**Proposed Solution:** A dual-model edge architecture:
- **Primary:** YOLOv8n (person/vehicle detection)
- **Secondary:** Lightweight SDNL CNN (adversarial patch anomaly detection)
- **Failsafe:** ESP32 → relay → gate lock (MQTT-triggered, <200ms end-to-end)
- **Dashboard:** Full-featured React + Node.js (live stream, attack gallery, analytics)

**Camera:** OAK-D Stereo Camera (with onboard MyriadX NPU for accelerated primary detection)

**Dataset:** APDE (94K images, 94 patch types, ICCV 2025) + APRICOT for evaluation

---

## 2. Novelty Claims (Patent + Publication)

| # | Claim | Type |
|---|---|---|
| N1 | Dual-concurrent model (primary detector + SDNL) on IoT-class edge device | System |
| N2 | SDNL trained on high-frequency spatial anomaly maps for physical patch detection | Method |
| N3 | Hardware-in-loop: CV detection → MQTT → ESP32 → physical gate lock (<200ms) | System |
| N4 | Stereo camera depth data as additional anomaly signal (depth inconsistency of flat patches) | Method |
| N5 | Real-time attack frame streaming to full-featured security dashboard with analytics | System |

---

## 3. Publication & Patent Strategy

```
Month 1-4  →  Build + Experiment + Collect Results
Month 4    →  Draft Provisional Patent Application (India + PCT)
Month 4-5  →  Write Paper (IEEE format, 8-12 pages)
Month 5    →  Submit to:
               PRIMARY:   IEEE Transactions on IoT (IF ~8.2)
               BACKUP:    IEEE Internet of Things Magazine
               CONFERENCE: USENIX Security / IEEE EDGE 2026
Month 6+   →  Full patent prosecution (engage patent attorney)
```

> **Why patent first?** Provisional patent filing establishes a priority date (12-month window) before public disclosure. This protects N1–N5 while allowing academic publication.

**Patentable Claims (Utility Patent — Indian Patent Office + PCT):**
1. A system for edge-based adversarial patch detection using concurrent dual-model inference with hardware failsafe coupling
2. Method for lightweight high-frequency spatial anomaly detection in real-time video streams from stereo cameras for adversarial patch identification
3. A cyber-physical security architecture where computer vision adversarial detection directly triggers microcontroller-level physical access control without cloud dependency

---

## 4. Milestone Timeline (20 Weeks Total)

| Milestone | Duration | Week | Deliverable | Status |
|---|---|---|---|---|
| **M0** | Planning + Lit Survey | Wk 1-2 | `docs/` files, literature table | ✅ DONE |
| **M1** | Environment Setup | Wk 2-3 | Python env, OAK-D SDK, dataset downloaded | ⬜ TODO |
| **M2** | Dataset Prep + EDA | Wk 3-4 | Cleaned dataset, train/val/test splits, frequency maps | ⬜ TODO |
| **M3** | SDNL Design + Training | Wk 5-7 | Trained SDNL model, ≥92% accuracy, <15ms on RPi | ⬜ TODO |
| **M4** | Primary Detector Integration | Wk 7-8 | YOLOv8n TFLite on OAK-D/RPi running at ≥20 FPS | ⬜ TODO |
| **M5** | Dual-Model Pipeline | Wk 8-9 | End-to-end: camera → detect → diagnose → alert | ⬜ TODO |
| **M6** | ESP32 Failsafe | Wk 9-10 | MQTT protocol, relay wired, gate lock tested | ⬜ TODO |
| **M7** | Dashboard — Backend | Wk 10-11 | Node.js + Socket.IO + REST API + SQLite | ⬜ TODO |
| **M8** | Dashboard — Frontend | Wk 11-13 | React live stream, alert gallery, analytics, override | ⬜ TODO |
| **M9** | Full Integration | Wk 13-14 | All components live; end-to-end test on physical setup | ⬜ TODO |
| **M10** | Benchmarking + Ablation | Wk 14-16 | Latency, accuracy, FPS, confusion matrix, ROC curves | ⬜ TODO |
| **M11** | Provisional Patent | Wk 16 | Filed with IP attorney / Patent office | ⬜ TODO |
| **M12** | Paper Writing | Wk 16-19 | Draft → review → final manuscript | ⬜ TODO |
| **M13** | Journal Submission | Wk 20 | IEEE Transactions on IoT (or USENIX Security) | ⬜ TODO |

---

## 5. Project Directory Structure

```
/home/prasannaa/Projects/p1/
├── docs/                        # All planning context files
│   ├── session_starter.md
│   ├── plans.md                 ← YOU ARE HERE
│   ├── architecture_plan.md
│   ├── build_plan.md
│   ├── progress.md
│   ├── feasibility.md
│   └── setup.md
│
├── data/                        # Datasets
│   ├── apde/                    # APDE 94K dataset (ICCV 2025)
│   ├── apricot/                 # APRICOT physical patches
│   ├── processed/               # Train/val/test splits
│   └── frequency_maps/         # DCT/FFT computed maps
│
├── models/                      # Trained model files
│   ├── sdnl/                    # Secondary Diagnostic Neural Layer
│   │   ├── sdnl_v1.pt           # PyTorch checkpoint
│   │   └── sdnl_v1.tflite       # Edge-optimized TFLite
│   └── primary/
│       └── yolov8n_infraedge.pt # Fine-tuned YOLOv8n
│
├── edge/                        # Edge node Python code
│   ├── pipeline.py              # Main dual-model pipeline
│   ├── sdnl_infer.py            # SDNL inference module
│   ├── yolo_infer.py            # Primary detector module
│   ├── mqtt_client.py           # MQTT alert publisher
│   └── stream.py                # Camera/video streaming
│
├── embedded/                    # ESP32 firmware
│   ├── esp32_failsafe/
│   │   ├── src/main.cpp         # Main firmware
│   │   └── platformio.ini       # PlatformIO config
│
├── dashboard/                   # Security dashboard
│   ├── backend/                 # Node.js + Express + Socket.IO
│   │   ├── server.js
│   │   ├── routes/
│   │   └── db/
│   └── frontend/                # React app
│       ├── src/
│       │   ├── App.jsx
│       │   ├── components/
│       │   └── pages/
│
├── training/                    # Model training scripts
│   ├── train_sdnl.py
│   ├── data_preprocessing.py
│   ├── augmentation.py
│   └── evaluate.py
│
├── experiments/                 # Experiment logs and results
│   ├── results/
│   └── plots/
│
└── paper/                       # Research paper drafts
    ├── draft_v1.tex
    ├── figures/
    └── references.bib
```

---

## 6. Success Metrics

| Metric | Target |
|---|---|
| SDNL Detection Accuracy | ≥ 92% on APDE test set |
| SDNL False Positive Rate | ≤ 5% on clean frames |
| SDNL Inference Time (RPi 4B) | ≤ 15 ms per ROI |
| End-to-End Latency (camera → ESP32 relay) | ≤ 200 ms |
| Primary Detector FPS (YOLOv8n, OAK-D) | ≥ 20 FPS |
| Dashboard Refresh Rate | Real-time (≤ 500ms via WebSocket) |
| Paper Acceptance | Top-tier IEEE journal or USENIX-class conference |
| Patent | Provisional filed before first public submission |
