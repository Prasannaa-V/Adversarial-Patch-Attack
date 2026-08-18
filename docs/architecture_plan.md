> ⚠️ **SUPERSEDED (2026-08-18).** This document describes the original **RPi 4B + OAK-D
> stereo-depth** design, which has been abandoned. The system is now a **single PC/laptop +
> ESP32** build with a generic USB/RTSP camera (no depth). For the current state see
> **`docs/session_starter.md`** and **`docs/progress.md`**; `docs/paper_sections.md` remains
> authoritative for the paper. Kept for historical reference only — do not implement from this file.

---

# 🏗️ ARCHITECTURE PLAN
## InfraEdge-Shield — System Architecture & Data Flow

---

## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHYSICAL ENVIRONMENT                                │
│                                                                             │
│   [Person/Vehicle]──[Adversarial Patch?]──▶  OAK-D Stereo Camera          │
└────────────────────────────────────────────────────────┬────────────────────┘
                                                         │ USB3 / MIPI
                    ┌────────────────────────────────────▼────────────────────┐
                    │            RASPBERRY PI 4B (Edge Compute Node)          │
                    │                                                          │
                    │  ┌────────────────────────────────────────────────┐     │
                    │  │         DUAL-MODEL PIPELINE (pipeline.py)      │     │
                    │  │                                                 │     │
                    │  │  Frame  ──▶  YOLOv8n (Primary Detector)       │     │
                    │  │              [TFLite INT8 / OAK-D NPU]        │     │
                    │  │                    │                            │     │
                    │  │            Detected ROI crops                  │     │
                    │  │                    │                            │     │
                    │  │                    ▼                            │     │
                    │  │  SDNL (Secondary Diagnostic Neural Layer)      │     │
                    │  │  ├── DCT high-freq map extraction              │     │
                    │  │  ├── MobileNetV3-Small CNN (<500K params)      │     │
                    │  │  └── Output: CLEAN | ATTACK (+ confidence)    │     │
                    │  │                    │                            │     │
                    │  └──────────────────┬─┴───────────────────────────┘     │
                    │                     │                                    │
                    │          ┌──────────┴──────────┐                        │
                    │          │ ATTACK DETECTED?     │                        │
                    │          └──────────┬──────────┘                        │
                    │            YES      │        NO                          │
                    │    ┌────────────────┘         └──────────────────┐      │
                    │    ▼                                              ▼      │
                    │  Publish MQTT alert              Normal pipeline flow   │
                    │  Stream frame to dashboard        (no action)            │
                    └────────┬───────────────────────────────────────────────-┘
                             │
               ┌─────────────┴──────────────────────────────────────┐
               │                                                      │
               ▼ MQTT (topic: infraedge/attack_alert)                ▼ WebSocket
    ┌─────────────────────┐                          ┌───────────────────────────┐
    │   ESP32 Failsafe    │                          │  Node.js Backend Server   │
    │   ────────────────  │                          │  ──────────────────────── │
    │   • Activate relay  │                          │  • Express REST API       │
    │   • Lock gate       │◀── MQTT ACK ──────────  │  • Socket.IO (real-time)  │
    │   • Flash LED       │                          │  • SQLite DB (log events) │
    │   • Sound buzzer    │                          │  • MJPEG stream forwarder │
    │   • Manual override │                          └──────────────┬────────────┘
    │     via dashboard   │                                         │ HTTP/WS
    └─────────────────────┘                          ┌─────────────▼────────────┐
                                                     │   React Dashboard        │
                                                     │   ──────────────────     │
                                                     │   • Live video stream    │
                                                     │   • Attack frame gallery │
                                                     │   • Alert log + timeline │
                                                     │   • Analytics charts     │
                                                     │   • Manual gate override │
                                                     │   • Confidence heatmap   │
                                                     └──────────────────────────┘
```

---

## 2. Component Deep-Dive

### 2.1 OAK-D Stereo Camera

| Feature | Specification |
|---|---|
| **Sensor** | RGB 12MP (IMX378) + dual mono stereo 1MP sensors |
| **Depth** | Active stereo, 7.5cm–1000cm range |
| **Onboard NPU** | Intel MyriadX VPU (4 TOPS) |
| **Interface** | USB 3.1 Gen1 to host (RPi) |
| **SDK** | DepthAI Python SDK (Luxonis) |

**Why OAK-D for this project:**
- MyriadX NPU can run YOLOv8n on-device, freeing RPi CPU for SDNL
- Stereo depth data is an **additional adversarial signal**: flat printed patches have abnormal depth profiles compared to real textured clothing/objects
- RGB + depth fusion provides a richer anomaly detection input

**OAK-D Pipeline Configuration:**
```python
# DepthAI pipeline: OAK-D runs primary detector onboard
pipeline = dai.Pipeline()
# ColorCamera → ImageManip → YOLODetectionNetwork (on MyriadX)
# Detections + RGB frame sent to RPi via XLink
# RPi receives (frame, bbox_list) and routes to SDNL
```

---

### 2.2 Primary Detector (YOLOv8n)

| Property | Value |
|---|---|
| **Model** | YOLOv8n (nano) |
| **Parameters** | ~3.2M |
| **Inference** | On OAK-D MyriadX NPU (preferred) OR RPi TFLite INT8 |
| **Input** | 640×640 RGB |
| **Output** | Bounding boxes + class + confidence |
| **Target classes** | `person`, `car`, `motorcycle`, `bus`, `truck` |
| **FPS target** | ≥20 FPS (OAK-D NPU), ≥12 FPS (RPi TFLite) |

**Fine-tuning:** YOLOv8n pre-trained on COCO; optionally fine-tuned on INRIA Person Dataset for access-control-specific appearance.

---

### 2.3 Secondary Diagnostic Neural Layer (SDNL)

**Concept:** The SDNL is a lightweight CNN that classifies whether a detected ROI contains an adversarial patch by analyzing its **high-frequency spatial content** — a signal that adversarial patches consistently exhibit regardless of their visual appearance.

```
ROI crop (from YOLOv8n bbox)
    │
    ▼
[Pre-processing]
    ├── Resize to 128×128
    ├── DCT (Discrete Cosine Transform) → high-freq component map
    ├── Laplacian filter → edge energy map
    └── Stack: [RGB | DCT_high | Laplacian] → 5-channel input
    │
    ▼
[MobileNetV3-Small backbone] (adapted, <500K params)
    ├── Depthwise separable conv blocks
    ├── SE (Squeeze-Excitation) attention
    └── Global Average Pooling
    │
    ▼
[Classifier Head]
    ├── FC(256) → ReLU → Dropout(0.3)
    └── FC(2) → Softmax → {CLEAN, ATTACK} + confidence score
```

**Training Configuration:**
```
Dataset     : APDE (94K, 94 patch types) + APRICOT
Split       : 70% train / 15% val / 15% test
Augmentation: Rotation ±30°, brightness ±40%, print noise, JPEG compression
Optimizer   : AdamW, lr=1e-3, weight_decay=1e-4
Scheduler   : CosineAnnealingLR
Loss        : CrossEntropy + Focal (γ=2) for class imbalance
Epochs      : 50 (early stopping patience=8)
Target      : ≥92% accuracy, ≤5% FPR
Export      : TFLite INT8 (via representative dataset calibration)
```

---

### 2.4 ESP32 Failsafe Node

```
ESP32 Firmware State Machine:

IDLE ──▶ MQTT subscribed to "infraedge/attack_alert"
  │
  │  On receive ATTACK message:
  │
  ├──▶ Activate Gate Lock Relay (GPIO 26, HIGH)
  ├──▶ Flash Red LED (GPIO 27, 5Hz for 10s)
  ├──▶ Sound Buzzer (GPIO 25, 880Hz for 2s)
  ├──▶ Publish ACK to "infraedge/attack_ack"
  └──▶ Start 30s lockout timer

  On receive RESET/OVERRIDE from dashboard:
  └──▶ Deactivate relay, return to IDLE

  On receive HEARTBEAT (every 5s from RPi):
  └──▶ Update watchdog timer (safety: if no heartbeat in 15s → safe-lock)
```

**MQTT Topics:**
| Topic | Publisher | Subscriber | Payload |
|---|---|---|---|
| `infraedge/attack_alert` | RPi | ESP32 | `{timestamp, confidence, frame_id}` |
| `infraedge/attack_ack` | ESP32 | RPi, Dashboard | `{timestamp, gate_status}` |
| `infraedge/override` | Dashboard | ESP32 | `{action: "unlock"/"lock"}` |
| `infraedge/heartbeat` | RPi | ESP32 | `{timestamp}` |
| `infraedge/status` | ESP32 | Dashboard | `{gate: "locked"/"unlocked", uptime}` |

**ESP32 Pin Mapping:**
| Pin | Function |
|---|---|
| GPIO 26 | Relay IN (gate lock activation) |
| GPIO 27 | Red LED (attack indicator) |
| GPIO 25 | Buzzer (PWM alert) |
| GPIO 32 | Green LED (system healthy) |
| GPIO 34 | Manual override button (INPUT) |

---

### 2.5 Node.js Backend

```
server.js
├── /api/events        GET  → Paginated attack event history
├── /api/events/:id    GET  → Single event with frame image
├── /api/gate          POST → Manual gate control (override)
├── /api/stats         GET  → Aggregated attack statistics
├── /stream            GET  → MJPEG live video proxy
└── Socket.IO events:
    ├── 'attack_detected' → Push new attack event to all clients
    ├── 'gate_status'     → Push ESP32 gate state changes
    └── 'system_health'   → Push RPi/ESP32 heartbeat status
```

**Database Schema (SQLite):**
```sql
CREATE TABLE attack_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME NOT NULL,
    frame_path  TEXT NOT NULL,
    confidence  REAL NOT NULL,
    gate_status TEXT NOT NULL,
    ack_time    DATETIME,
    operator    TEXT,
    notes       TEXT
);

CREATE TABLE system_health (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME NOT NULL,
    component   TEXT NOT NULL,  -- 'rpi', 'esp32', 'camera'
    status      TEXT NOT NULL,  -- 'online', 'offline', 'degraded'
    fps         REAL,
    latency_ms  REAL
);
```

---

### 2.6 React Dashboard (Full-Featured)

**Pages / Views:**
```
App
├── /                  → Main Dashboard
│   ├── Live stream (MJPEG/WebRTC)
│   ├── Gate status indicator (LOCKED / UNLOCKED)
│   ├── System health bar (RPi FPS, SDNL latency, ESP32 ping)
│   └── Recent alerts sidebar (last 10 events)
│
├── /alerts            → Attack Event Gallery
│   ├── Grid of flagged frames with timestamps
│   ├── Filter by date/confidence/status
│   └── Click to view full frame + confidence heatmap
│
├── /analytics         → Statistics & Charts
│   ├── Attack frequency over time (line chart)
│   ├── Confidence score distribution (histogram)
│   ├── Detection rate vs FPS performance (dual-axis)
│   └── System uptime / availability gauge
│
└── /settings          → System Configuration
    ├── SDNL confidence threshold slider
    ├── Gate lockout duration
    ├── Alert notification preferences
    └── Manual gate override button
```

---

## 3. Data Flow (End-to-End)

```
T=0ms    OAK-D captures RGB frame + depth map
T=15ms   YOLOv8n on MyriadX returns bbox list
T=16ms   RPi receives (frame, bboxes) via USB XLink
T=18ms   ROI crops extracted per bbox
T=30ms   SDNL analyzes ROI (high-freq map → CNN → output)
T=31ms   [IF CLEAN] → discard, continue
T=31ms   [IF ATTACK] →
         ├── Publish MQTT to ESP32
         ├── Save frame to /frames/ directory
         ├── POST event to dashboard backend via HTTP
         └── Socket.IO push to connected dashboard clients
T=85ms   ESP32 receives MQTT, activates relay (gate locks)
T=86ms   ESP32 publishes ACK
T=100ms  Dashboard receives Socket.IO event, updates UI
T=150ms  Dashboard shows live alert with frame
         (End-to-end: ~150-200ms)
```

---

## 4. Stereo Depth as Auxiliary Adversarial Signal (Novel Extension)

**Hypothesis:** Adversarial patches are flat, printed 2D surfaces. When the stereo camera computes depth maps, a genuine textured object (clothing, skin) will show **continuous, natural depth gradients**. A flat printed patch on that object will show a **localized region of uniform depth** inconsistent with the surrounding 3D surface normal expectations.

**Implementation (Phase 6 extension):**
```python
def depth_anomaly_score(depth_roi: np.ndarray) -> float:
    """
    Computes a flatness anomaly score for an ROI depth patch.
    High score → suspiciously flat → possible printed patch.
    """
    gradient_magnitude = np.sqrt(
        cv2.Sobel(depth_roi, cv2.CV_64F, 1, 0)**2 +
        cv2.Sobel(depth_roi, cv2.CV_64F, 0, 1)**2
    )
    return 1.0 - (np.mean(gradient_magnitude) / np.max(gradient_magnitude + 1e-6))
```

This depth signal is **fused** with the SDNL confidence score:
```
final_score = 0.7 * sdnl_confidence + 0.3 * depth_anomaly_score
```

This depth-fusion approach is a **novel contribution** not present in any existing literature.

---

## 5. Security Considerations

| Concern | Mitigation |
|---|---|
| MQTT broker compromise | TLS/SSL on Mosquitto broker; client certificates |
| ESP32 firmware tampering | ESP32 Secure Boot + Flash Encryption enabled |
| Dashboard unauthorized access | JWT authentication on all API routes |
| RPi physical access | Encrypted storage; read-only root filesystem |
| Camera feed interception | RTSP over TLS; local network only (no internet exposure) |
| DoS on detection system | Watchdog on RPi pipeline; ESP32 defaults to LOCKED on heartbeat loss |
