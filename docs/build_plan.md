> ⚠️ **SUPERSEDED (2026-08-18).** This document describes the original **RPi 4B + OAK-D
> stereo-depth** design, which has been abandoned. The system is now a **single PC/laptop +
> ESP32** build with a generic USB/RTSP camera (no depth). For the current state see
> **`docs/session_starter.md`** and **`docs/progress.md`**; `docs/paper_sections.md` remains
> authoritative for the paper. Kept for historical reference only — do not implement from this file.

---

# 🔨 BUILD PLAN
## InfraEdge-Shield — Step-by-Step Implementation Guide

---

## PHASE 1: Environment Setup

> See `setup.md` for detailed installation commands.

### 1.1 Host Machine (Development PC)
```bash
# Clone / initialize project
cd /home/prasannaa/Projects/p1
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy scipy matplotlib seaborn tqdm
pip install depthai blobconverter  # OAK-D SDK
pip install paho-mqtt flask-socketio sqlite3
```

### 1.2 Raspberry Pi 4B Setup
```bash
# On RPi (Raspberry Pi OS 64-bit, Bookworm)
sudo apt update && sudo apt install -y python3-pip libopenblas-dev
pip3 install tflite-runtime ultralytics paho-mqtt depthai
# Install Mosquitto MQTT broker
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
```

### 1.3 Dataset Download
```bash
mkdir -p /home/prasannaa/Projects/p1/data
cd /home/prasannaa/Projects/p1/data

# APDE Dataset (ICCV 2025, 94K images)
git clone https://github.com/Gandolfczjh/APDE apde/

# APRICOT Dataset
wget https://apricot.cs.columbia.edu/apricot_v1.1.zip
unzip apricot_v1.1.zip -d apricot/
```

---

## PHASE 2: Dataset Preparation

### 2.1 Data Preprocessing Script

**File:** `training/data_preprocessing.py`

**Tasks:**
- Load APDE + APRICOT annotations
- Extract ROI crops from bounding boxes (person/vehicle + patch region)
- Resize all crops to 128×128
- Compute frequency maps:
  ```python
  # DCT high-frequency extraction
  dct_map = cv2.dct(np.float32(gray_roi))
  high_freq_mask = np.zeros_like(dct_map)
  high_freq_mask[64:, 64:] = dct_map[64:, 64:]  # Top-right quadrant = high-freq
  
  # Laplacian edge energy
  laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
  ```
- Stack into 5-channel tensor: `[R, G, B, DCT_high, Laplacian]`
- Create binary labels: `0=CLEAN`, `1=ATTACK`
- Save as HDF5 or directory structure for PyTorch `ImageFolder`

### 2.2 Train/Val/Test Split
```python
# Stratified 70/15/15 split preserving patch type distribution
from sklearn.model_selection import train_test_split
```

### 2.3 Class Balance Check
- Expected: ~50% clean, ~50% attack (from APDE structure)
- If imbalanced: use Focal Loss (γ=2) + oversampling clean class

---

## PHASE 3: SDNL Training

### 3.1 Model Architecture

**File:** `training/train_sdnl.py`

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SDNL(nn.Module):
    def __init__(self, num_classes=2, in_channels=5):
        super().__init__()
        # Adapt MobileNetV3-Small for 5-channel input
        self.backbone = models.mobilenet_v3_small(pretrained=False)
        # Replace first conv layer
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
```

### 3.2 Training Loop Key Config
```python
config = {
    "epochs": 50,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 8,           # Early stopping
    "loss": "focal",         # FocalLoss(gamma=2)
    "scheduler": "cosine",
    "device": "cuda",        # Train on GPU
    "input_size": (128, 128),
    "in_channels": 5,
}
```

### 3.3 Export to TFLite INT8
```python
# After training, convert to TFLite for edge deployment
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("models/sdnl/sdnl_v1_saved")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open("models/sdnl/sdnl_v1.tflite", "wb") as f:
    f.write(tflite_model)
```

### 3.4 Evaluation Metrics to Report
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve
- Confusion matrix
- Inference time on RPi 4B (average over 1000 runs)
- Per-patch-type detection rate (from APDE's 94 types)

---

## PHASE 4: Primary Detector Integration

### 4.1 OAK-D DepthAI Pipeline
**File:** `edge/yolo_infer.py`

```python
import depthai as dai
import cv2

def build_oak_pipeline(model_blob_path: str) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    
    # Color camera
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setFps(30)
    
    # YOLO detection network (runs on MyriadX NPU)
    det = pipeline.create(dai.node.YoloDetectionNetwork)
    det.setBlobPath(model_blob_path)
    det.setConfidenceThreshold(0.5)
    det.setNumClasses(80)
    det.setCoordinateSize(4)
    det.setAnchors([...])  # YOLOv8 anchors
    
    # Link nodes
    cam.preview.link(det.input)
    
    # XLink outputs to host (RPi)
    xout_frame = pipeline.create(dai.node.XLinkOut)
    xout_frame.setStreamName("rgb")
    cam.preview.link(xout_frame.input)
    
    xout_det = pipeline.create(dai.node.XLinkOut)
    xout_det.setStreamName("detections")
    det.out.link(xout_det.input)
    
    return pipeline
```

### 4.2 Depth Stream (Auxiliary Signal)
```python
# Add stereo depth to the pipeline
left = pipeline.create(dai.node.MonoCamera)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right = pipeline.create(dai.node.MonoCamera)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
left.out.link(stereo.left)
right.out.link(stereo.right)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)
```

---

## PHASE 5: Dual-Model Pipeline

**File:** `edge/pipeline.py`

```python
import threading
import queue
from sdnl_infer import SDNLInference
from yolo_infer import build_oak_pipeline
from mqtt_client import MQTTAlertClient

class DualModelPipeline:
    def __init__(self, config):
        self.sdnl = SDNLInference("models/sdnl/sdnl_v1.tflite")
        self.mqtt = MQTTAlertClient(config["broker_host"])
        self.roi_queue = queue.Queue(maxsize=10)
        self.threshold = config["sdnl_threshold"]  # default: 0.85
    
    def run(self):
        # Thread 1: OAK-D frame + detection reader
        # Thread 2: SDNL inference worker
        # Thread 3: MQTT + dashboard publisher
        ...
    
    def on_detection(self, frame, bbox, depth_roi):
        rgb_roi = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        score = self.sdnl.infer(rgb_roi)
        
        # Fuse with depth anomaly signal
        d_score = depth_anomaly_score(depth_roi)
        final_score = 0.7 * score + 0.3 * d_score
        
        if final_score >= self.threshold:
            self.trigger_alert(frame, final_score, bbox)
    
    def trigger_alert(self, frame, confidence, bbox):
        # 1. Save frame
        frame_path = save_attack_frame(frame, bbox)
        # 2. Publish MQTT
        self.mqtt.publish_alert(confidence, frame_path)
        # 3. Notify dashboard
        requests.post("http://localhost:3000/api/events", json={
            "confidence": confidence,
            "frame_path": frame_path,
            "timestamp": datetime.utcnow().isoformat()
        })
```

---

## PHASE 6: ESP32 Failsafe Firmware

**File:** `embedded/esp32_failsafe/src/main.cpp`

```cpp
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// Pin definitions
#define RELAY_PIN     26
#define LED_RED       27
#define BUZZER_PIN    25
#define LED_GREEN     32
#define OVERRIDE_BTN  34

const char* MQTT_TOPIC_ALERT     = "infraedge/attack_alert";
const char* MQTT_TOPIC_ACK       = "infraedge/attack_ack";
const char* MQTT_TOPIC_OVERRIDE  = "infraedge/override";
const char* MQTT_TOPIC_HEARTBEAT = "infraedge/heartbeat";
const char* MQTT_TOPIC_STATUS    = "infraedge/status";

// State machine
enum SystemState { IDLE, LOCKED, OVERRIDE };
SystemState currentState = IDLE;

unsigned long lastHeartbeat = 0;
const unsigned long HEARTBEAT_TIMEOUT = 15000; // 15s

void onAttackAlert(String payload) {
    // Parse JSON payload
    DynamicJsonDocument doc(256);
    deserializeJson(doc, payload);
    float confidence = doc["confidence"];
    
    // Activate failsafe
    digitalWrite(RELAY_PIN, HIGH);     // Lock gate
    tone(BUZZER_PIN, 880, 2000);       // 880Hz for 2s
    currentState = LOCKED;
    
    // Publish ACK with gate status
    String ack = "{\"gate_status\":\"LOCKED\",\"confidence\":" + 
                  String(confidence) + "}";
    mqttClient.publish(MQTT_TOPIC_ACK, ack.c_str());
}

void checkHeartbeat() {
    // Safety: if no heartbeat from RPi in 15s → lock gate
    if (millis() - lastHeartbeat > HEARTBEAT_TIMEOUT) {
        digitalWrite(RELAY_PIN, HIGH);
        currentState = LOCKED;
    }
}
```

**PlatformIO Config:**
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
lib_deps =
    knolleary/PubSubClient@^2.8
    bblanchon/ArduinoJson@^6.21
build_flags =
    -DMQTT_MAX_PACKET_SIZE=512
```

---

## PHASE 7: Dashboard Backend

**File:** `dashboard/backend/server.js`

```javascript
const express = require('express');
const { Server } = require('socket.io');
const mqtt = require('mqtt');
const sqlite3 = require('sqlite3');
const multer = require('multer');
const cors = require('cors');

const app = express();
const httpServer = require('http').createServer(app);
const io = new Server(httpServer, { cors: { origin: '*' } });

// MQTT bridge: subscribe to ESP32 status and push to dashboard clients
const mqttClient = mqtt.connect('mqtt://localhost:1883');
mqttClient.subscribe(['infraedge/attack_ack', 'infraedge/status']);
mqttClient.on('message', (topic, message) => {
    io.emit('mqtt_event', { topic, payload: JSON.parse(message) });
});

// REST API routes
app.get('/api/events', async (req, res) => { ... });
app.post('/api/events', async (req, res) => {
    // Called by RPi pipeline when attack is detected
    const { confidence, frame_path, timestamp } = req.body;
    // Insert to SQLite
    // Emit Socket.IO event to all dashboard clients
    io.emit('attack_detected', { confidence, frame_path, timestamp });
    res.json({ status: 'ok' });
});
app.post('/api/gate', async (req, res) => {
    // Manual override: publish MQTT to ESP32
    const { action } = req.body;
    mqttClient.publish('infraedge/override', JSON.stringify({ action }));
    res.json({ status: 'ok' });
});

httpServer.listen(3000);
```

---

## PHASE 8: Dashboard Frontend

**Key React Components:**
```
src/
├── components/
│   ├── LiveStream.jsx          # MJPEG player + bbox overlay
│   ├── GateStatus.jsx          # LOCKED/UNLOCKED indicator with animation
│   ├── AlertFeed.jsx           # Real-time incoming alerts sidebar
│   ├── AttackGallery.jsx       # Grid of flagged frames
│   ├── ConfidenceHeatmap.jsx   # Grad-CAM visualization overlay
│   ├── AnalyticsChart.jsx      # Chart.js line/bar charts
│   ├── SystemHealth.jsx        # FPS, latency, uptime gauges
│   └── OverrideButton.jsx      # Manual gate control with confirmation
│
├── pages/
│   ├── Dashboard.jsx           # Main page
│   ├── Alerts.jsx              # Full alert history
│   ├── Analytics.jsx           # Statistics page
│   └── Settings.jsx            # Configuration page
│
└── hooks/
    ├── useSocket.js            # Socket.IO connection hook
    └── useMQTTStatus.js        # ESP32 status subscription
```

---

## PHASE 9: Benchmarking Experiments

### Experiments to Run & Report:

| Experiment | Metric | Expected |
|---|---|---|
| SDNL accuracy vs patch types | Per-type F1 across 94 APDE types | ≥88% mean |
| SDNL inference time (RPi 4B) | ms per ROI (TFLite INT8) | ≤15 ms |
| Full pipeline FPS | Frames per second end-to-end | ≥20 FPS |
| End-to-end latency | Camera → ESP32 relay trigger | ≤200 ms |
| False positive rate | On clean COCO frames | ≤5% |
| Depth fusion improvement | SDNL-only vs SDNL+Depth | Δ F1 >3% |
| Ablation: w/o high-freq maps | Accuracy vs RGB-only input | Δ ≥8% |
| Attack under different lighting | ASR at 500 lux vs 50 lux | Characterize |

### Paper Figures to Generate:
1. System architecture block diagram (Tikz/draw.io)
2. SDNL accuracy vs training epochs
3. ROC curve (SDNL vs PatchZero vs Jedi baselines)
4. Per-patch-type detection heatmap (94 types)
5. Latency breakdown (camera → SDNL → MQTT → relay)
6. Dashboard screenshot (annotated)
7. Physical testbed photograph
8. Depth anomaly signal visualization (novel)
