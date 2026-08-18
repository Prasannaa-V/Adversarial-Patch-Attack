# ⚙️ SETUP GUIDE
## InfraEdge-Shield — Complete Environment Setup Instructions

---

## 0. Prerequisites Checklist

Before starting, confirm you have:
- [ ] Development PC (Linux/Windows) with Python 3.11+
- [ ] Raspberry Pi 4B (4GB RAM recommended) with Raspberry Pi OS 64-bit (Bookworm)
- [ ] OAK-D Lite / OAK-D / Intel RealSense D435i camera
- [ ] ESP32 DevKit V1
- [ ] MicroSD card ≥32GB (for RPi)
- [ ] USB 3.0 cable (for OAK-D)
- [ ] Relay module, LEDs, buzzer, jumper wires

---

## 1. Development PC Setup

### 1.1 Create Project & Virtual Environment

```bash
cd /home/prasannaa/Projects/p1
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 1.2 Install Core ML Dependencies

```bash
# PyTorch (with CUDA 11.8 — adjust for your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If no GPU (use CPU or Google Colab for training):
# pip install torch torchvision torchaudio

# Ultralytics (YOLOv8)
pip install ultralytics

# Scientific computing
pip install numpy scipy matplotlib seaborn scikit-learn tqdm h5py

# Computer vision
pip install opencv-python pillow
```

### 1.3 Install OAK-D SDK (DepthAI)

```bash
pip install depthai
pip install blobconverter  # Convert ONNX to OAK-D blob format

# Verify installation
python3 -c "import depthai; print(depthai.__version__)"
```

### 1.4 Install Edge/IoT Dependencies

```bash
pip install paho-mqtt  # MQTT client
pip install requests flask  # API communication
pip install tflite-runtime  # TFLite inference (for local testing)

# TensorFlow (for TFLite conversion only)
pip install tensorflow
```

### 1.5 Install Dashboard Development Tools

```bash
# Node.js (install via nvm for version management)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20
node --version  # Should show v20.x.x

# Dashboard backend dependencies
cd /home/prasannaa/Projects/p1/dashboard/backend
npm init -y
npm install express socket.io mqtt sqlite3 multer cors jsonwebtoken bcryptjs dotenv

# Dashboard frontend
cd /home/prasannaa/Projects/p1/dashboard/frontend
npx create-react-app . --template typescript
npm install socket.io-client chart.js react-chartjs-2 axios react-router-dom
```

### 1.6 Save Requirements

```bash
cd /home/prasannaa/Projects/p1
pip freeze > requirements.txt
```

---

## 2. Dataset Setup

### 2.1 APDE Dataset (ICCV 2025) — Primary Dataset

```bash
cd /home/prasannaa/Projects/p1/data

# Option A: Clone from GitHub
git clone https://github.com/Gandolfczjh/APDE apde/
cd apde && ls  # Inspect structure

# Expected structure:
# apde/
# ├── images/          (94K images)
# ├── annotations/     (COCO-format JSON)
# ├── patch_types/     (94 patch categories)
# └── README.md
```

### 2.2 APRICOT Dataset — Physical Printed Patches

```bash
cd /home/prasannaa/Projects/p1/data

# Download from Columbia University
wget https://apricot.cs.columbia.edu/data/apricot_v1.1.zip
unzip apricot_v1.1.zip -d apricot/
ls apricot/

# Expected structure:
# apricot/
# ├── images/           (real-world photographed scenes with patches)
# ├── annotations.json  (patch bounding boxes)
# └── patches/          (individual patch images)
```

### 2.3 INRIA Person Dataset — Clean Person Frames

```bash
cd /home/prasannaa/Projects/p1/data

wget http://pascal.inrialpes.fr/data/human/INRIA_person_dataset.zip
unzip INRIA_person_dataset.zip -d inria/
```

### 2.4 COCO val2017 — Clean Scene Frames (FPR Testing)

```bash
cd /home/prasannaa/Projects/p1/data

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d coco/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d coco/
```

---

## 3. Raspberry Pi 4B Setup

### 3.1 Flash RPi OS
1. Download **Raspberry Pi OS 64-bit (Bookworm)** from raspberrypi.com
2. Flash to MicroSD using **Raspberry Pi Imager**
3. Enable SSH in Imager settings (set hostname, username, password, WiFi)

### 3.2 Connect and Update

```bash
# SSH into RPi (or use direct keyboard/monitor)
ssh prasannaa@raspberrypi.local

# Update system
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-pip python3-venv libopenblas-dev
sudo apt install -y libatlas-base-dev libhdf5-dev
```

### 3.3 Install Python Dependencies on RPi

```bash
python3 -m venv ~/venv_infraedge
source ~/venv_infraedge/bin/activate

# TFLite runtime (optimized for RPi ARM64)
pip install tflite-runtime

# Ultralytics (for YOLOv8 TFLite inference helper)
pip install ultralytics --no-deps
pip install numpy opencv-python paho-mqtt requests

# DepthAI SDK (OAK-D)
pip install depthai
```

### 3.4 Install and Configure Mosquitto MQTT Broker

```bash
sudo apt install -y mosquitto mosquitto-clients

# Configure Mosquitto
sudo nano /etc/mosquitto/mosquitto.conf
```

Add these lines to mosquitto.conf:
```
listener 1883 0.0.0.0
allow_anonymous true
log_dest file /var/log/mosquitto/mosquitto.log
```

```bash
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Test broker
mosquitto_pub -t "test" -m "hello"
mosquitto_sub -t "test" -v
```

### 3.5 Connect OAK-D to RPi

```bash
# Add udev rules for OAK-D USB access
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Connect OAK-D via USB 3.0 cable
# Test connection
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
# Should show [DeviceInfo(MXID=...)]
```

### 3.6 Enable RPi Performance Mode

```bash
# Max CPU frequency for inference performance
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable swap (reduces SD card writes during inference)
sudo dphys-swapfile swapoff
sudo systemctl disable dphys-swapfile
```

---

## 4. ESP32 Setup (PlatformIO)

### 4.1 Install PlatformIO

```bash
# Install PlatformIO CLI
pip install platformio

# Or use VS Code PlatformIO extension (recommended)
# Install VS Code → Extensions → search "PlatformIO IDE"
```

### 4.2 Initialize ESP32 Project

```bash
cd /home/prasannaa/Projects/p1/embedded

# Initialize PlatformIO project
pio project init --board esp32dev --ide vscode
```

### 4.3 Configure platformio.ini

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps =
    knolleary/PubSubClient@^2.8
    bblanchon/ArduinoJson@^6.21
build_flags =
    -DMQTT_MAX_PACKET_SIZE=512
    -DCORE_DEBUG_LEVEL=3
```

### 4.4 WiFi + MQTT Configuration

Create `embedded/esp32_failsafe/include/config.h`:

```cpp
#ifndef CONFIG_H
#define CONFIG_H

// WiFi credentials
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// MQTT broker (RPi IP address on local network)
#define MQTT_BROKER   "192.168.1.XXX"  // Replace with RPi's IP
#define MQTT_PORT     1883
#define MQTT_CLIENT_ID "infraedge_esp32"

// Pin definitions
#define RELAY_PIN     26
#define LED_RED       27
#define BUZZER_PIN    25
#define LED_GREEN     32
#define OVERRIDE_BTN  34

#endif
```

### 4.5 Upload Firmware

```bash
cd /home/prasannaa/Projects/p1/embedded/esp32_failsafe
pio run --target upload
pio device monitor  # View serial output
```

---

## 5. Hardware Wiring Diagram

```
ESP32 DevKit V1
─────────────────────────────────────────────────
ESP32 3.3V  ──────────────────── VCC (Relay, LEDs)
ESP32 GND   ──────────────────── GND (all components)
ESP32 GPIO26 ──── IN  ──── [5V Relay Module]
                               COM ──── 12V Gate Lock +
                               NO  ──── Gate Lock -
                               (NC = normally closed for safety)

ESP32 GPIO27 ──── [330Ω] ──── Anode ──── [Red LED] ──── GND
ESP32 GPIO32 ──── [330Ω] ──── Anode ──── [Green LED] ──── GND
ESP32 GPIO25 ──── [Buzzer +] ──── GND (PWM driven)
ESP32 GPIO34 ──── [Pushbutton] ──── GND (INPUT_PULLUP)
─────────────────────────────────────────────────

OAK-D Camera:
USB 3.0 ──── Raspberry Pi 4B USB 3.0 port (blue)
(Powered via USB, no separate power needed for OAK-D Lite)

Power:
RPi 4B ──── USB-C 5V/3A Power Supply
Gate Lock ──── 12V/1A DC Adapter (separate)
Relay shared ground with ESP32
```

---

## 6. Network Configuration

```
Local Network (192.168.1.0/24):
────────────────────────────────
RPi 4B         192.168.1.100  (static IP)  ← MQTT broker, edge pipeline
ESP32          192.168.1.101  (DHCP)       ← MQTT client, failsafe
Dev PC         192.168.1.XXX  (DHCP)       ← Dashboard access, SSH
Dashboard      http://192.168.1.100:3000   ← React frontend
MQTT Broker    mqtt://192.168.1.100:1883   ← Mosquitto

# Set static IP on RPi:
sudo nano /etc/dhcpcd.conf
# Add:
# interface eth0
# static ip_address=192.168.1.100/24
# static routers=192.168.1.1
# static domain_name_servers=8.8.8.8
```

---

## 7. Quick Test Checklist

After setup, verify each component:

```bash
# 1. OAK-D camera test
cd /home/prasannaa/Projects/p1
python3 -c "
import depthai as dai
devices = dai.Device.getAllAvailableDevices()
print(f'OAK-D devices found: {len(devices)}')
"

# 2. MQTT broker test (from dev PC)
mosquitto_pub -h 192.168.1.100 -t "infraedge/test" -m "hello"
mosquitto_sub -h 192.168.1.100 -t "infraedge/test" -v

# 3. SDNL TFLite test (after model is ready)
python3 -c "
import tflite_runtime.interpreter as tflite
interp = tflite.Interpreter('models/sdnl/sdnl_v1.tflite')
interp.allocate_tensors()
print('SDNL model loaded successfully')
print('Input shape:', interp.get_input_details()[0]['shape'])
"

# 4. Dashboard backend test
cd /home/prasannaa/Projects/p1/dashboard/backend
node server.js &
curl http://localhost:3000/api/events
# Should return: {"events": [], "total": 0}

# 5. ESP32 MQTT test (check serial monitor for messages)
mosquitto_pub -h 192.168.1.100 -t "infraedge/attack_alert" \
  -m '{"timestamp":"2026-08-17T10:00:00Z","confidence":0.97,"frame_id":"frame_001"}'
# ESP32 should: lock relay, flash LED, sound buzzer, send ACK
```

---

## 8. Common Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| OAK-D not detected | USB 2.0 port used | Use blue USB 3.0 port on RPi |
| OAK-D not detected | udev rules missing | Re-run udev setup commands in §3.5 |
| MQTT connection refused | Mosquitto not listening on 0.0.0.0 | Check mosquitto.conf listener setting |
| ESP32 not connecting to WiFi | Wrong SSID/password in config.h | Update config.h and re-flash |
| TFLite model not found | Wrong path | Use absolute paths; check `pwd` |
| RPi CPU at 100% | Running both models on CPU | Confirm OAK-D NPU is handling primary detector |
| Dashboard shows no events | RPi pipeline not posting to API | Check RPi network config and backend URL |
