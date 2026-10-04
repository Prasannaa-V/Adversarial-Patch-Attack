# DIAGRAM SOURCE CODE & PROMPTS

This document contains all standalone diagram definitions, UML source code, and visual prompts for the Capstone Project.

---

## Section 9: Architecture Eraser.io Prompt

```
title Smart Access Control Defense System Architecture

Camera [icon: camera, color: teal] {
  label: "Camera\n(USB / IP)\nRGB @ 30 FPS"
}

PC [icon: monitor, color: blue] {
  label: "Edge Inference Host\n(PC / Laptop)"

  YOLOv8n [icon: cpu, color: indigo] {
    label: "YOLOv8n\nObject Detector"
  }

  SDNL [icon: chip, color: purple] {
    label: "SDNL CNN\n4-ch ONNX CPU\n<500K params · 1.66ms"
  }

  MQTT_PUB [icon: radio, color: cyan] {
    label: "MQTT Publisher\n+ Frame Store"
  }

  Backend [icon: server, color: green] {
    label: "Node.js Backend\n+ Mosquitto Broker\n+ SQLite DB"
  }

  Dashboard [icon: layout-dashboard, color: emerald] {
    label: "React Dashboard\nLive Stream · Alerts\nAnalytics · Override"
  }
}

ESP32 [icon: microchip, color: orange] {
  label: "ESP32\nFailsafe Controller"

  Relay [icon: toggle-right, color: red] {
    label: "Relay → Gate Lock\nRelay Output"
  }

  Indicators [icon: bell, color: amber] {
    label: "Red LED · Buzzer\nGreen LED · Button"
  }
}

GateLock [icon: lock, color: red] {
  label: "Gate Lock\n12V Relay-controlled"
}

// Connections
Camera --> YOLOv8n: "RGB Frames"
YOLOv8n --> SDNL: "ROI Crops"
SDNL --> MQTT_PUB: "ATTACK ≥ 0.70"
MQTT_PUB --> Backend: "HTTP POST /api/events"
Backend --> Dashboard: "Socket.IO events"
MQTT_PUB --> ESP32: "MQTT smartgate/attack_alert (QoS 1)"
ESP32 --> Backend: "MQTT smartgate/attack_ack"
ESP32 <--> Backend: "heartbeat / status"
Dashboard --> Backend: "POST /api/gate (override)"
Relay --> GateLock: "Relay Output LOW → LOCK (Fail-Secure)"
```

---

## Section 10.1: Standard UML Use Case Diagram (PlantUML)

```plantuml
@startuml
skinparam dpi 300
left to right direction
skinparam shadowing false
skinparam packageStyle rectangle

' Visual Styling (Blue Ovals matching reference style)
skinparam usecase {
  BackgroundColor #60A5FA
  BorderColor #2563EB
  FontColor White
  FontSize 12
  FontStyle bold
}

skinparam actor {
  BackgroundColor #F3F4F6
  BorderColor #374151
  FontColor #1F2937
  FontSize 12
}

skinparam arrow {
  Color #4B5563
  FontSize 10
}

' --- Left Actors ---
actor "Camera Sensor" as CAM
actor "Security Operator" as OP

' --- Right Actors ---
actor "ESP32 Controller" as ESP
actor "System Admin" as AD

rectangle "Smart Infrastructure Access Control System" {

  ' Primary Use Cases
  usecase "Detect Objects\n& Patch" as UC_Detect
  usecase "Diagnose Spatial\nAnomaly" as UC_Diagnose
  usecase "Actuate Gate\nLock Relay" as UC_Lock
  usecase "Monitor Live\nDashboard" as UC_Stream
  usecase "Trigger Gate\nOverride" as UC_Override
  usecase "Tune Detection\nThreshold" as UC_Threshold

  ' Secondary / Included Use Cases
  usecase "Extract 2D DCT\nSpectrum Map" as UC_DCT
  usecase "Verify HMAC\nSignature" as UC_HMAC
  usecase "Emit Alarm\nLED & Buzzer" as UC_Alarm
  usecase "Authenticate\nOperator (JWT)" as UC_Auth
  usecase "Enforce Watchdog\nAuto-Lock" as UC_Watchdog

  ' Relationships & Dependencies
  UC_Diagnose ..> UC_DCT : <<include>>
  UC_Diagnose ..> UC_Detect : <<include>>
  
  UC_Lock ..> UC_HMAC : <<include>>
  UC_Lock ..> UC_Alarm : <<include>>
  UC_Watchdog ..> UC_Lock : <<extend>>

  UC_Override ..> UC_Auth : <<include>>
}

' --- Left Actor Associations ---
CAM -- UC_Detect
CAM -- UC_Diagnose
OP -- UC_Stream
OP -- UC_Override

' --- Right Actor Associations ---
ESP -- UC_Lock
ESP -- UC_Watchdog
AD -- UC_Threshold
AD -- UC_Stream

@enduml
```

### **Use Case Diagram Narrative Description**

The Use Case Diagram defines the operational boundary interactions for the Smart Infrastructure Access Control System across four external actors and six core functional use cases. The Camera Sensor actor initiates visual threat evaluation by interacting with the primary object detection and spatial anomaly diagnosis use cases (`Detect Objects & Patch` and `Diagnose Spatial Anomaly`), which mandatorily include 2D Discrete Cosine Transform spectral extraction (`Extract 2D DCT Spectrum Map`) to construct 4-channel tensors. Upon confirming an adversarial patch anomaly exceeding the tuned confidence threshold (0.70), the ESP32 Controller actor interacts with the hardware actuation use case (`Actuate Gate Lock Relay`) to maintain a de-energized LOW relay state (fail-secure posture), mandatorily including Zero-Trust payload signature verification (`Verify HMAC Signature`) and local alarm signaling (`Emit Alarm LED & Buzzer`). Additionally, the watchdog auto-lock use case (`Enforce Watchdog Auto-Lock`) extends relay actuation if the 15-second host heartbeat is lost. Human actors interact through dedicated security interface use cases: the Security Operator interacts with live dashboard monitoring (`Monitor Live Dashboard`) and manual override controls (`Trigger Gate Override`), which mandatorily include operator JWT bearer token verification (`Authenticate Operator`), while the System Administrator interfaces with system tuning controls (`Tune Detection Threshold`) to modify detection sensitivity parameters at runtime.

---

## Section 10.2: Class Diagram (PlantUML)

```plantuml
@startuml
skinparam dpi 300
skinparam packageStyle rectangle
skinparam classAttributeIconSize 0
skinparam nodesep 15
skinparam ranksep 20

' Class Styling
skinparam class {
  BackgroundColor White
  HeaderBackgroundColor #F4F5F7
  BorderColor #333333
  FontSize 10
}

package "Data Models" {
  class AttackEvent {
    + id : int
    + timestamp : datetime
    + confidence : float
    + frame_path : str
    + gate_status : GateStatus
    + to_dict() : dict
  }

  enum GateStatus {
    LOCKED
    UNLOCKED
    OVERRIDE
  }

  class BBox {
    + x1, y1, x2, y2 : int
    + label : str
    + confidence : float
    + to_roi(frame) : ndarray
  }

  AttackEvent -right-> GateStatus
  AttackEvent -down-> BBox
}

package "Inference Pipeline" {
  class DetectionPipeline {
    - detector : YOLODetector
    - classifier : SDNLClassifier
    - mqtt : MQTTClient
    - threshold : float
    + run() : void
    + stop() : void
    - process_frame(frame) : void
    - on_attack(conf, frame_id) : void
  }

  class YOLODetector {
    - model : YOLO
    - cap : VideoCapture
    + get_frame() : FrameBundle
    + detect(frame) : list[BBox]
    + release() : void
  }

  class MQTTClient {
    - client : paho.Client
    - broker : str
    + connect() : void
    + publish_alert(conf, frame_id) : void
    + publish_heartbeat(fps) : void
    + subscribe(topic, cb) : void
  }

  class SDNLClassifier {
    - session : ort.InferenceSession
    - threshold : float
    + infer(roi) : tuple[str, float]
    + preprocess(roi) : ndarray
    - build_4ch_tensor(bgr) : ndarray
  }

  DetectionPipeline -down-> YOLODetector
  DetectionPipeline -down-> MQTTClient
  DetectionPipeline -down-> SDNLClassifier

  YOLODetector -[hidden]right-> MQTTClient
  MQTTClient -[hidden]right-> SDNLClassifier
}

package "ESP32 Firmware" {
  class ESP32Controller {
    - state : FSMState
    - mqtt : PubSubClient
    - watchdogMs : ulong
    + setup() : void
    + loop() : void
    + onAttackAlert() : void
    + onOverride() : void
    + checkWatchdog() : void
    - activateRelay() : void
  }

  enum FSMState {
    IDLE
    LOCKED
    OVERRIDE
  }

  ESP32Controller -right-> FSMState
}

package "Dashboard & Web" {
  class BackendServer {
    - app : Express
    - io : SocketIO
    - db : SQLite
    + start() : void
    + postEvent(e) : void
    + emitAlert(e) : void
    + overrideGate(action) : void
  }

  class ReactDashboard {
    - gateStatus : GateStatus
    - alerts : AttackEvent[]
    + renderStream() : JSX
    + renderGallery() : JSX
    + renderAnalytics() : JSX
    + sendOverride(a) : void
  }

  BackendServer -down-> ReactDashboard : "Socket.IO"
}

"Data Models" -[hidden]right-> "Inference Pipeline"
"Inference Pipeline" -[hidden]down-> "ESP32 Firmware"
"ESP32 Firmware" -right[hidden]-> "Dashboard & Web"

DetectionPipeline ..> AttackEvent : "creates"
YOLODetector ..> BBox : "produces"

MQTTClient -down-> ESP32Controller : "MQTT smartgate/attack_alert"
ESP32Controller -right-> BackendServer : "MQTT smartgate/attack_ack"
BackendServer -up-> MQTTClient : "MQTT smartgate/override"

@enduml
```

---

## Section 10.3: Sequence Diagram (PlantUML)

```plantuml
@startuml
skinparam shadowing false
skinparam monochrome false
skinparam SequenceMessageAlignment center
skinparam defaultFontName "Segoe UI", "Helvetica", "Arial", sans-serif
skinparam defaultFontSize 12
skinparam NoteBackgroundColor #FEF3C7
skinparam NoteBorderColor #F59E0B
skinparam DividerBackgroundColor #F3F4F6
skinparam DividerBorderColor #E5E7EB

autonumber "[00]"

participant "Camera" as CAM #1ABC9C
participant "YOLOv8n" as YOLO #3498DB
participant "SDNL\nClassifier" as SDNL #8b5cf6
participant "MQTT\nBroker" as MQTT #f59e0b
participant "ESP32" as ESP #ef4444
participant "Node.js\nBackend" as NODE #22c55e
participant "Dashboard" as UI #14b8a6

== Frame Capture & Primary Detection ==
CAM -> YOLO : RGB Frame (USB / RTSP)
activate YOLO
YOLO -> YOLO : YOLOv8n inference\n→ Bounding Boxes
YOLO -> SDNL : ROI crops (128×128)
deactivate YOLO
activate SDNL

== SDNL Inference & HMAC Attestation (1.66 ms) ==
SDNL -> SDNL : Build 4-ch tensor [RGB + 2D DCT]
SDNL -> SDNL : ONNX CPU inference
note right of SDNL
  confidence = 0.94
  ATTACK confirmed
end note

par Parallel Dispatch
  SDNL ->> MQTT : PUBLISH smartgate/attack_alert\n{conf: 0.94, frame_id: 42, hmac_sig: "..."}
  SDNL ->> NODE : Async HTTP POST /api/events
end
deactivate SDNL

== Physical Failsafe Actuation (< 60 ms total) ==
activate MQTT
MQTT ->> ESP : DELIVER smartgate/attack_alert (QoS 1)
deactivate MQTT
activate ESP

ESP -> ESP : Verify HMAC-SHA256 signature (mbedTLS)
ESP -> ESP : State: IDLE → LOCKED\nRELAY_PIN LOW → Gate Physically Locked\nRed LED 5Hz + Buzzer 880Hz
ESP ->> MQTT : PUBLISH smartgate/attack_ack {gate: LOCKED, hmac_sig: "..."}
deactivate ESP

activate MQTT
MQTT ->> NODE : DELIVER smartgate/attack_ack
deactivate MQTT

== Dashboard Notification (< 150 ms total) ==
activate NODE
NODE -> NODE : INSERT event & ack to SQLite (UNIQUE frame_id)
NODE -> UI : Socket.IO: attack_detected
deactivate NODE
activate UI
UI -> UI : Show alert banner + 🔴 LOCKED badge
deactivate UI

== Watchdog Heartbeat (Background) ==
loop Every 5 seconds
  SDNL ->> MQTT : PUBLISH smartgate/heartbeat (with HMAC)
  MQTT ->> ESP : DELIVER smartgate/heartbeat
  ESP -> ESP : Reset 15s watchdog timer
end

note over ESP
  If no heartbeat in 15s:
  auto-lock (de-energized LOW = LOCKED)
end note

@enduml
```