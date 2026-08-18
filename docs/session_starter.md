# 🚀 SESSION STARTER
## Real-Time Detection and Mitigation of Physical Adversarial Patch Attacks in Edge-Based Smart Infrastructure

> **Read this file first at the start of every new session.**
> It tells you exactly where you are, what was last done, and what to do next.

---

## 📌 Project Identity

| Field | Value |
|---|---|
| **Project Title** | Real-Time Detection and Mitigation of Physical Adversarial Patch Attacks in Edge-Based Smart Infrastructure |
| **Short Name** | InfraEdge-Shield |
| **Goal** | Build a working system + publish research paper + file patent |
| **Primary Publication Target** | IEEE Transactions on IoT (Journal) OR USENIX Security (Conference) |
| **Patent Strategy** | **Provisional Patent FIRST** → then publish |
| **Project Root** | `/home/prasannaa/Projects/p1/` |
| **Docs Root** | `/home/prasannaa/Projects/p1/docs/` |

---

## ⚙️ Confirmed Hardware & Software Choices

| Component | Choice | Notes |
|---|---|---|
| **Edge Compute Node** | Raspberry Pi 4B (primary) | Runs YOLOv8n + SDNL concurrently |
| **Camera** | OAK-D (Stereo + Neural Processing) OR Intel RealSense D435i | OAK-D preferred: onboard MyriadX NPU offloads detection |
| **Failsafe MCU** | ESP32 | MQTT subscriber → triggers relay/gate lock on attack |
| **Dataset** | APDE (94K, 94 patch types, ICCV 2025) + APRICOT | Newest available; no self-collection needed |
| **Dashboard** | Full-featured React + Node.js | Live stream + attack gallery + analytics + manual override |
| **Patent** | File provisional BEFORE first public presentation | Protects dual-model + hardware-in-loop novelty |

---

## 🗂️ Context Files Index

| File | Location | Purpose |
|---|---|---|
| **session_starter.md** ← YOU ARE HERE | `docs/session_starter.md` | Quick-start every session |
| **plans.md** | `docs/plans.md` | Full roadmap, milestones, timeline |
| **architecture_plan.md** | `docs/architecture_plan.md` | System diagrams, data flow, API contracts |
| **build_plan.md** | `docs/build_plan.md` | Step-by-step implementation guide |
| **progress.md** | `docs/progress.md` | Live task tracker (update every session) |
| **feasibility.md** | `docs/feasibility.md` | Technical, hardware, publication feasibility |
| **setup.md** | `docs/setup.md` | Environment setup, wiring, dependency install |

---

## 🔴 Current Phase

```
[PHASE 0] ✅ Planning & Literature Survey COMPLETE
[PHASE 1] ⬜ Environment Setup & Dataset Preparation  ← NEXT
[PHASE 2] ⬜ Secondary Diagnostic Neural Layer (SDNL) Training
[PHASE 3] ⬜ Dual-Model Edge Integration
[PHASE 4] ⬜ ESP32 Hardware Failsafe
[PHASE 5] ⬜ Full-Featured Dashboard
[PHASE 6] ⬜ System Integration & Benchmarking
[PHASE 7] ⬜ Provisional Patent Filing
[PHASE 8] ⬜ Paper Writing & Journal Submission
```

---

## 🧠 Core System Concept (One Paragraph)

A stereo camera (OAK-D) streams video to a Raspberry Pi 4B. **YOLOv8n** (primary detector) runs at ~30 FPS and detects persons/vehicles. For each detected ROI, a **Secondary Diagnostic Neural Layer (SDNL)** — a custom lightweight CNN (<500K params) — analyzes high-frequency spatial anomalies in parallel to identify adversarial patch signatures. On detection, the Pi sends an MQTT alert to an **ESP32**, which autonomously locks the gate via a relay. Simultaneously, the flagged frame is streamed to a **React + Node.js** security dashboard for human review and logging.

---

## 🆕 What To Do Next Session

1. Check `progress.md` for current task status
2. Start Phase 1: run `setup.md` instructions
3. Download APDE dataset from: https://github.com/Gandolfczjh/APDE
4. Set up Python virtual environment per `setup.md`

---

## 📞 Key References (Always Available)

- APDE Benchmark (ICCV 2025): https://arxiv.org/abs/2508.00649
- Jedi (CVPR 2023): https://openaccess.thecvf.com/content/CVPR2023
- YOLOv8 Docs: https://docs.ultralytics.com
- OAK-D DepthAI SDK: https://docs.luxonis.com
- APRICOT Dataset: https://apricot.cs.columbia.edu
