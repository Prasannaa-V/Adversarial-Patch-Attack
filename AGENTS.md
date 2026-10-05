# 🤖 AGENTS.md — InfraEdge-Shield Project Guide

> **Primary Onboarding Document for AI Coding Agents (OpenCode, Antigravity, Claude Code, etc.)**  
> *Project:* Real-Time Detection & Mitigation of Physical Adversarial Patch Attacks in Edge-Based Smart Infrastructure  
> *Repository Root:* `/home/prasannaa/Projects/p1`  
> *Last Updated:* October 2026

---

## 🎯 1. Project Purpose & Scope

This project implements an end-to-end, real-time edge security system that defends physical access control infrastructure (automated gates, secure entryways, parking barriers) against **physical adversarial patch attacks**. 

When an attacker presents a printed adversarial patch (such as an APRICOT patch) designed to induce target blindness or misclassification in primary object detectors:
1. **Edge Detection:** A local dual-model pipeline (YOLOv8n + 4-channel SDNL CNN) detects the patch in **< 2 ms** on CPU without cloud dependence.
2. **Physical Mitigation:** An edge microcontroller (ESP32) triggers a hardware relay failsafe (**de-energized LOW = LOCKED** fail-secure posture) within **~60 ms**, physically denying entry.
3. **Operator Oversight:** A real-time Node.js + React 19 security dashboard streams live annotated video, issues sound/visual alerts, logs HMAC-authenticated audit events, and provides operator override controls.
4. **Academic/IP Output:** Target submission to top-tier venues (IEEE Transactions on IoT / Computers & Security) and an Indian provisional patent filing.

---

## 📚 2. Master Documentation Guide (`docs/` Index)

Before writing code or making architectural assumptions, **consult the documentation directory**. Note that some older design documents describe an abandoned hardware architecture.

### ✅ Active & Authoritative Documentation (Read These!)

| Document | Path | Purpose & Content |
|---|---|---|
| **Session Starter** | [`docs/session_starter.md`](file:///home/prasannaa/Projects/p1/docs/session_starter.md) | **Start here.** Verified as-built architecture, hardware specifications, data flow, and critical gotchas. |
| **Progress & Status** | [`docs/progress.md`](file:///home/prasannaa/Projects/p1/docs/progress.md) | Current verified component checklist, empirical benchmark numbers, and task history. |
| **Paper Draft & Theory** | [`docs/paper_sections.md`](file:///home/prasannaa/Projects/p1/docs/paper_sections.md) | **Authoritative academic specification.** Full Abstract, Introduction, Problem Statement, Methodology, Mathematical Formulations, Threat Model, System Architecture, and References. Essential for writing papers or patents. |
| **Diagram Specifications** | [`docs/diagram_sources.md`](file:///home/prasannaa/Projects/p1/docs/diagram_sources.md) | Standalone diagram source code (Eraser.io, Mermaid, PlantUML) for system architecture, hardware wiring, and sequence flows. |
| **Project Plans & Roadmap** | [`docs/plans.md`](file:///home/prasannaa/Projects/p1/docs/plans.md) | High-level roadmap, milestone breakdown, and threat model aligned with the current as-built system. |

### ⚠️ Superseded / Historical Documents (Do NOT implement from these)

These files document an earlier, abandoned prototype iteration that used a Raspberry Pi 4B and OAK-D stereo-depth camera. They are preserved purely for design history and carry `SUPERSEDED` banners:

- [`docs/build_plan.md`](file:///home/prasannaa/Projects/p1/docs/build_plan.md) — *Superseded RPi 4B / OAK-D build plan.*
- [`docs/architecture_plan.md`](file:///home/prasannaa/Projects/p1/docs/architecture_plan.md) — *Superseded 5-channel depth-fusion architecture.*
- [`docs/setup.md`](file:///home/prasannaa/Projects/p1/docs/setup.md) — *Superseded RPi toolchain setup steps.*
- [`docs/feasibility.md`](file:///home/prasannaa/Projects/p1/docs/feasibility.md) — *Superseded depth sensor feasibility analysis.*

---

## 🏗️ 3. As-Built System Architecture

```
[ Camera (USB/RTSP/Dummy) ]
            │ (RGB Frames)
            ▼
┌────────────────────────────────────────────────────────┐
│ Edge Host Workstation (PC / Laptop)                    │
│                                                        │
│  1. Primary Detector: YOLOv8n (OpenCV)                 │
│     Extracts candidate ROIs (person / vehicle)         │
│                                                        │
│  2. Secondary Diagnostic Neural Layer (SDNL):          │
│     4-Channel MobileNetV3-Small ONNX (<500K params)    │
│     Input: RGB (3-ch) + 2D DCT High-Freq Energy (1-ch) │
│     Latency: ~1.99 ms on CPU · Threshold: 0.703        │
│                                                        │
│  3. Multithreaded Pipeline (inference/pipeline.py):    │
│     Reader Thread ──► SDNL Worker ──► Alert Publisher  │
│                                                        │
│  4. Zero-Trust MQTT Client (HMAC-SHA256 signed):       │
│     Publishes attack_alert & heartbeats to Mosquitto   │
└────────────┬─────────────────────────────┬─────────────┘
             │ MQTT QoS1                   │ HTTP POST (HMAC)
             │ `smartgate/attack_alert`    │ & Socket.IO
             ▼                             ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│ ESP32 Failsafe Controller │ │ Dashboard Stack          │
│ - Hardware Relay          │ │ - Node/Express Backend   │
│ - LOW = LOCKED (Failsafe) │ │   JWT Auth, SQLite DB,   │
│ - 15s Watchdog auto-lock  │ │   Live Frame Annotation  │
│ - 30s Auto-release        │ │ - React 19 + Vite UI     │
│ - Buzzer & Manual Button  │ │   Dark mode, Live Alert, │
└───────────────────────────┘ │   Acknowledge, Gate Ctrl │
                              └──────────────────────────┘
```

---

## 📁 4. Codebase Directory Map

| Directory | Key Files | Description |
|---|---|---|
| [`inference/`](file:///home/prasannaa/Projects/p1/inference/) | [`detector.py`](file:///home/prasannaa/Projects/p1/inference/detector.py)<br>[`sdnl.py`](file:///home/prasannaa/Projects/p1/inference/sdnl.py)<br>[`pipeline.py`](file:///home/prasannaa/Projects/p1/inference/pipeline.py)<br>[`mqtt_client.py`](file:///home/prasannaa/Projects/p1/inference/mqtt_client.py) | **Core edge inference engine.** Camera ingestion, YOLOv8n candidate extraction, 4-channel DCT feature generation, SDNL ONNX inference, HMAC-signed MQTT publishing. |
| [`dashboard/backend/`](file:///home/prasannaa/Projects/p1/dashboard/backend/) | [`server.js`](file:///home/prasannaa/Projects/p1/dashboard/backend/server.js)<br>`package.json` | **Central server & API bridge.** Express REST API, JWT auth (`/api/auth/login`), Socket.IO WebSocket, SQLite persistent event database (`UNIQUE(frame_id)`), `/api/live-frame` route. |
| [`dashboard/frontend/`](file:///home/prasannaa/Projects/p1/dashboard/frontend/) | [`src/App.jsx`](file:///home/prasannaa/Projects/p1/dashboard/frontend/src/App.jsx)<br>`vite.config.js` | **Operator dashboard.** React 19 + Vite, Lucide icons, live video feed with bounding boxes, event audit log, acknowledge incident, gate unlock override. |
| [`esp32/firmware/`](file:///home/prasannaa/Projects/p1/esp32/firmware/) | [`src/main.cpp`](file:///home/prasannaa/Projects/p1/esp32/firmware/src/main.cpp)<br>`platformio.ini` | **Hardware failsafe.** ESP32 C++ firmware. Listens to MQTT topic `smartgate/attack_alert`, enforces fail-secure relay lock, 15 s watchdog, non-blocking buzzer. |
| [`training/`](file:///home/prasannaa/Projects/p1/training/) | [`train_sdnl.py`](file:///home/prasannaa/Projects/p1/training/train_sdnl.py)<br>[`benchmark_sdnl.py`](file:///home/prasannaa/Projects/p1/training/benchmark_sdnl.py)<br>[`prepare_clean_independent_dataset.py`](file:///home/prasannaa/Projects/p1/training/prepare_clean_independent_dataset.py) | **Model training & evaluation.** Clean COCO vs APRICOT patch dataset preparation, 4-channel PyTorch training, ONNX export, and rigorous validation benchmark. |
| [`experiments/results/`](file:///home/prasannaa/Projects/p1/experiments/results/) | `eval_report.txt`<br>`pr_curve.png`<br>`roc_curve.png`<br>`confusion_matrix.png` | **Empirical results.** Generated evaluation curves, precision-recall curve, confusion matrix, and latency benchmarks. |
| [`config/`](file:///home/prasannaa/Projects/p1/config/) | [`config.yaml`](file:///home/prasannaa/Projects/p1/config/config.yaml) | **Global configuration.** Camera endpoints, MQTT broker & topic names, SDNL model path, thresholds, and timeout intervals. |
| [`models/`](file:///home/prasannaa/Projects/p1/models/) | `sdnl/sdnl_v1.onnx`<br>`sdnl/sdnl_v1.pth` | Exported ONNX model weights (<500K parameters) and PyTorch checkpoints. |

---

## ⚡ 5. How to Run, Test, and Verify

### Prerequisites & Python Environment
The workspace uses a local Python virtual environment in `venv/`:
```bash
cd /home/prasannaa/Projects/p1
source venv/bin/activate
```

### 1. Run Evaluation Benchmark (PR Curve, ROC, Confusion Matrix)
To verify the SDNL model performance on the held-out test split:
```bash
venv/bin/python training/benchmark_sdnl.py --eval-only
```
*Outputs are saved to `experiments/results/`.*

### 2. Start Dashboard Backend (Port 3000)
```bash
cd /home/prasannaa/Projects/p1/dashboard/backend
npm start
```
- Listens on `http://localhost:3000`
- REST APIs: `/api/events`, `/api/health`, `/api/gate`, `/api/auth/login`, `/api/live-frame`
- Default Operator Credentials:
  - **Username:** `admin`
  - **Password:** `password123`

### 3. Start Dashboard Frontend (Port 5173)
```bash
cd /home/prasannaa/Projects/p1/dashboard/frontend
npm run dev
```
- Available at `http://localhost:5173`

### 4. Run Live Edge Inference Pipeline
```bash
cd /home/prasannaa/Projects/p1
source venv/bin/activate
python inference/pipeline.py
```
*(If no webcam is connected, set `camera_index: "dummy"` or test with image sample in `config/config.yaml`).*

---

## 📊 6. Current Verified Benchmark Numbers

From the held-out test set evaluation (`experiments/results/eval_report.txt`):

| Metric | Result | Target/Standard |
|---|---|---|
| **Optimal Threshold** | **0.703** | F1-optimised on validation split |
| **Accuracy** | **98.12%** | > 95.0% |
| **Precision** | **99.22%** | (Only 1 false positive in 160 test samples) |
| **Recall** | **98.46%** | High sensitivity to physical attack |
| **F1-Score** | **0.9884** | Robust balance |
| **AUC-PR** | **0.9952** | High area under precision-recall curve |
| **AUC-ROC** | **0.9954** | Near-perfect discriminability |
| **ONNX CPU Latency** | **1.99 ms (P95: 3.71 ms)** | Edge-ready (< 15 ms target) |
| **Relay Failsafe Actuation** | **60.37 ms** | Physical cutoff < 200 ms |

---

## 🚀 7. Next Actions & Roadmap for Incoming Agents

If continuing the project towards paper publication, patent filing, or final deployment:

1. **Ablation Study (Key Academic Gap):**
   - Create an evaluation script comparing:
     - Baseline 3-channel (RGB only) vs. Proposed 4-channel (RGB + 2D DCT) SDNL classifier.
     - Document how the DCT frequency spectrum specifically suppresses false positives on texture-rich clean objects.
2. **Paper Formatting:**
   - Migrate content from [`docs/paper_sections.md`](file:///home/prasannaa/Projects/p1/docs/paper_sections.md) into the IEEE Conference/Transaction LaTeX template.
   - Include the generated plots from `experiments/results/pr_curve.png`, `roc_curve.png`, and `confusion_matrix.png`.
3. **Patent Specification:**
   - Review Novelty Points in [`docs/paper_sections.md`](file:///home/prasannaa/Projects/p1/docs/paper_sections.md) and [`docs/session_starter.md`](file:///home/prasannaa/Projects/p1/docs/session_starter.md) (Zero-Trust HMAC physical fail-secure closed-loop edge actuation).
4. **Physical Demo Setup:**
   - Print sample APRICOT patch targets (from `data/apricot/`) on physical paper.
   - Verify webcam live detection and ESP32 physical relay lock in real-world ambient room lighting.

---

*This document was generated to guarantee seamless agent handoffs across OpenCode, Antigravity, and other coding assistants.*
