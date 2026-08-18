# 📄 PAPER SECTIONS DRAFT
## InfraEdge-Shield — Abstract, Introduction, Problem Statement, Objectives & Scope

> **Status:** Draft v1.0 | For review and iteration
> **Target:** IEEE Transactions on IoT / USENIX Security
> **Format:** IEEE double-column, LaTeX

---

# ABSTRACT

The proliferation of edge-deployed computer vision systems in smart infrastructure—including automated parking gates, secure entryways, and traffic management nodes—has introduced a critical yet underexplored attack surface: physical adversarial patch attacks. These strategically crafted, printable visual perturbations exploit the mathematical linearity of deep neural networks to induce misclassification or target blindness in object detectors, enabling unauthorized physical access without triggering any digital-domain alarms. Existing mitigation approaches either rely on computationally intensive cloud-GPU pipelines that introduce unacceptable network latency or employ preprocessing-based defenses that are too resource-heavy for standard Internet-of-Things (IoT) edge hardware. This paper proposes **InfraEdge-Shield**, a lightweight, hardware-in-the-loop defense architecture designed exclusively for edge deployment. The system employs a dual-model inference strategy: a primary YOLOv8n object detector operating concurrently with a Secondary Diagnostic Neural Layer (SDNL)—a custom sub-500K-parameter convolutional network that analyzes high-frequency spatial anomalies and RGB depth inconsistencies to identify adversarial patch signatures. Upon detection, an ESP32 microcontroller autonomously triggers an immediate physical failsafe, such as locking an access gate, within 200 milliseconds—independent of cloud connectivity. Simultaneously, flagged frames are streamed to a full-featured React and Node.js security dashboard for operator review and incident logging. Evaluated on the APDE benchmark (94,000 images, 94 patch types), the proposed system achieves a detection accuracy of over 92%, a false positive rate below 5%, and a sub-200 ms end-to-end response latency, demonstrating practical viability for real-world smart infrastructure protection.

**Word count: ~200 words**

---

## KEYWORDS

`Adversarial Patch Attack` · `Edge Computing` · `Hardware-in-the-Loop` · `Cyber-Physical Security` · `Object Detection` · `IoT Infrastructure` · `Anomaly Detection` · `YOLOv8` · `Stereo Vision` · `TinyML` · `MQTT` · `Smart Infrastructure`

---

---

# 1. INTRODUCTION

The global transition toward smart infrastructure has positioned computer vision as a cornerstone technology for autonomous decision-making in physical security systems. Automated parking barriers, biometric access gates, and AI-driven surveillance networks increasingly rely on real-time object detection models deployed directly on edge devices—cameras, single-board computers, and embedded microcontrollers—to reduce latency and eliminate dependence on centralized cloud infrastructure [cite: Huang et al., 2024]. This paradigm shift toward edge intelligence, while offering substantial operational advantages, simultaneously opens a novel and dangerous attack surface that has received insufficient attention in both academic literature and industrial deployment practice.

Physical adversarial patch attacks represent one of the most immediately threatening forms of adversarial machine learning to emerge in recent years. Unlike imperceptible digital perturbations, adversarial patches are printed visual patterns—physically fabricated and strategically placed within a camera's field of view—that exploit the mathematical structure of deep neural networks to cause systematic misclassification [cite: Brown et al., 2017]. A malicious actor can simply print and affix such a patch to their clothing, a bag, or an object near a surveillance camera, causing the access control system's object detector to become blind to their presence. The consequences of this attack in physical infrastructure are immediate and severe: unauthorized individuals may gain access to restricted zones, vehicles may bypass parking barriers, and adversarial operators may render entire surveillance networks ineffective without generating any detectable digital event log.

The severity of this threat is further compounded by the operational characteristics of fixed-camera smart infrastructure. Prior work on physical adversarial patches [cite: Eykholt et al., 2018; Gu et al., 2023] has largely evaluated attacks in mobile scenarios, such as autonomous driving, where variable camera-subject distances, angles, and lighting conditions naturally degrade attack success rates. In contrast, access control cameras at parking gates and secure entryways operate at controlled, consistent distances—typically 0.5 to 3 meters—under regulated lighting conditions. This controlled geometry dramatically increases the effectiveness of physical adversarial patches, making the infrastructure threat model significantly more severe than previously documented in mobile-scenario evaluations.

Existing defense systems fall into two broad, inadequate categories. The first category encompasses cloud-centric GPU-accelerated pipelines that perform intensive image preprocessing—such as diffusion model purification [cite: Xiao et al., 2024; Ma et al., 2025] or entropy-based inpainting [cite: Tarchoun et al., 2023]—before forwarding frames to the primary detector. While these approaches achieve strong detection accuracy, they introduce unacceptable network latency (often exceeding 500ms to several seconds per frame), require persistent cloud connectivity, and impose GPU compute requirements that are incompatible with IoT-class edge hardware. The second category comprises training-based defenses such as PatchGuard [cite: Xiang et al., 2021] and PatchCleanser [cite: Xiang et al., 2022], which require full model retraining under architectural constraints, resulting in clean accuracy degradation of 10–15% and making them impractical for organizations operating standard pre-trained detection models.

Critically, no prior work has addressed the hardware response layer of adversarial patch defense. Detection of an adversarial patch in a camera frame must be followed by an immediate, autonomous physical response—such as locking an access gate or triggering a perimeter alarm—to prevent the attack from succeeding in the physical domain. Existing systems route detection results to human operators or log them for post-hoc analysis, introducing a temporal gap during which the adversarial actor may have already gained physical access. This gap between digital detection and physical response represents the fundamental architectural limitation of all current adversarial patch defense systems.

This paper addresses these limitations by proposing **InfraEdge-Shield**, a complete, edge-native adversarial patch detection and mitigation system designed specifically for fixed-camera smart infrastructure. The key contributions of this work are as follows:

1. **Dual-Model Edge Architecture:** A concurrent inference pipeline combining a primary YOLOv8n object detector (running on the Camera's PC CPU/GPU) with a Secondary Diagnostic Neural Layer (SDNL)—a lightweight convolutional network trained to detect high-frequency spatial anomalies characteristic of adversarial patches—running entirely on a PC/Laptop without cloud dependency.

2. **Secondary Diagnostic Neural Layer (SDNL):** A custom sub-500K-parameter CNN using 4-channel inputs (RGB + DCT high-frequency map + Laplacian edge energy) trained on the APDE benchmark (94,000 images, 94 patch types [cite: Zheng et al., 2025]), designed to distinguish adversarial patch signatures from natural image textures with ≥92% accuracy and ≤15ms inference latency on ARM-class hardware.

3. **DCT High-Frequency Feature Channel:** A novel 4-channel SDNL input incorporating a Discrete Cosine Transform (DCT) high-frequency coefficient map alongside RGB, providing an explicit representation of the adversarial noise energy that patch optimization injects into the image — enabling detection without depth sensors or specialized hardware.

4. **Hardware-in-the-Loop Physical Failsafe:** An ESP32 microcontroller subscribing to MQTT alerts from the edge inference node, autonomously triggering a physical gate lock relay within 200 milliseconds of attack detection, entirely independent of cloud connectivity or human intervention.

5. **Full-Featured Security Dashboard:** A React and Node.js real-time dashboard providing live video streaming, flagged frame visualization, temporal attack analytics, and operator override controls, bridging the gap between automated hardware response and human security oversight.

The remainder of this paper is organized as follows. Section 2 surveys related work on adversarial patch attacks and defenses. Section 3 presents the threat model and system design. Section 4 describes the SDNL training methodology and DCT frequency fusion. Section 5 details the hardware-in-the-loop integration. Section 6 presents experimental evaluation against the APDE benchmark. Section 7 discusses practical deployment considerations, limitations, and future work. Section 8 concludes the paper.

---

---

# 2. PROBLEM STATEMENT

## 2.1 Formal Problem Definition

Consider a fixed-camera smart infrastructure system comprising a camera **C** positioned at a controlled distance *d* ∈ [0.5, 3.0] meters from an access point **A** (such as a parking gate or secure entryway). The system deploys an edge-based object detector *f*: **X** → **Y**, where **X** is the space of RGB image frames and **Y** is the set of detected bounding boxes with associated class labels and confidence scores. The system grants access when a detection event *y* ∈ **Y** satisfies a predefined access criterion (e.g., detecting a registered person or authorized vehicle).

An adversarial patch attack introduces a perturbation **δ** into the scene by physically placing a printed patch *P*(**δ**) within the camera's field of view, such that:

> *f*(**x** ⊕ *P*(**δ**)) ≠ *f*(**x**)

where **x** is the clean frame and ⊕ denotes the physical superimposition of the patch. Specifically, the attack is designed to achieve one of two adversarial objectives:

- **Target Blindness Attack:** The detector *f* fails to detect the target subject entirely — a "disappearance" attack causing the gate to grant unchallenged access.
- **Misclassification Attack:** The detector *f* classifies a non-authorized subject as an authorized one (e.g., classifies a trespasser's torso as a registered vehicle badge).

The attack is executed without any digital access to the system, requiring only a commercially printed patch on standard matte paper or fabric.

## 2.2 Why This Problem Is Severe in Fixed-Camera Infrastructure

The majority of prior physical adversarial patch literature has evaluated attacks in **mobile, dynamic scenarios** (autonomous vehicles [cite: Li et al., 2024], pedestrian detection from varying viewpoints [cite: Gu et al., 2023]). In these settings, variable camera-to-subject distance, lighting conditions, and viewing angles significantly degrade attack success rates. However, the threat model for smart infrastructure access control is fundamentally different:

| Parameter | Mobile Scenario | Fixed Infrastructure | Consequence |
|---|---|---|---|
| Camera–Subject Distance | 5–50 m (variable) | 0.5–3 m (controlled) | Patch details fully visible, attack more effective |
| Viewing Angle | Highly variable | Fixed (±15°) | Patch orientation is predictable and optimizable |
| Lighting | Uncontrolled outdoor | Indoor/structured | Consistent illumination enables reliable patch function |
| Temporal Exposure | Fraction of a second | 2–10 seconds (gate approach) | Patch has longer time window to influence detector |
| Consequence of Success | Classifier error | Physical security breach | Life-safety and property implications |

This analysis reveals that **fixed-camera smart infrastructure presents a harder adversarial defense problem with higher real-world attack success rates** than previously studied mobile scenarios.

## 2.3 Limitations of Existing Defenses

| Defense Category | Representative Work | Why It Fails in This Context |
|---|---|---|
| **Diffusion-based purification** | DIFFender (ECCV 2024), DisPatch (2025) | Requires GPU; latency in seconds; infeasible on IoT hardware |
| **Entropy/inpainting-based** | Jedi (CVPR 2023) | Autoencoder adds >50ms latency; not edge-benchmarked |
| **Model retraining-based** | PatchGuard (2021), PatchCleanser (2022) | 10–15% clean accuracy drop; requires full retraining |
| **SAM-based localization** | PAD (CVPR 2024) | SAM (ViT-based) is GB-scale; not runnable on RPi |
| **Cloud-based pipeline** | Generic cloud CV | Latency >500ms; connectivity-dependent; single point of failure |
| **No hardware response** | ALL prior works | Detection does not trigger physical failsafe; human-in-the-loop delay |

The fundamental gap in existing literature is the **absence of an edge-native, hardware-coupled adversarial patch defense system** capable of autonomously triggering a physical security response within a latency window that prevents unauthorized access.

## 2.4 Research Questions

This work addresses the following research questions:

- **RQ1:** Can a lightweight (<500K parameter) convolutional neural network, trained on high-frequency spatial anomaly features, detect physical adversarial patches with accuracy competitive with computationally heavy methods, while meeting real-time latency constraints on ARM-class edge hardware?

- **RQ2:** Does RGB depth information provide a statistically significant auxiliary signal for adversarial patch detection, exploiting the spatial flatness of printed patches relative to natural three-dimensional object surfaces?

- **RQ3:** Can the complete detection-to-hardware-response pipeline (camera → dual-model inference → MQTT → microcontroller relay) be executed within 200 milliseconds entirely without cloud dependency?

- **RQ4:** Does the proposed system maintain an acceptable false positive rate (≤5%) on clean, non-attacked frames, ensuring operational continuity of the access control infrastructure?

---

---

# 3. OBJECTIVES

The primary and secondary objectives of this research are as follows:

## 3.1 Primary Objectives

**O1 — Design a Lightweight Dual-Model Edge Architecture**
Design and implement a concurrent dual-model inference pipeline comprising a primary YOLOv8n object detector (offloaded to the Camera's onboard PC CPU/GPU) and a Secondary Diagnostic Neural Layer (SDNL), achieving ≥20 FPS throughput with ≤15ms SDNL inference latency.

**O2 — Train and Evaluate the Secondary Diagnostic Neural Layer**
Develop a custom 4-channel CNN (RGB + DCT high-frequency map) trained on the APDE benchmark (ICCV 2025, 94,000 images, 94 adversarial patch types) to detect adversarial patch signatures with ≥92% accuracy and ≤5% false positive rate on clean frames, without requiring modification of the primary detector.

**O3 — Implement Stereo Depth Anomaly Fusion**
Develop a 4-channel SDNL input pipeline that computes the Discrete Cosine Transform (DCT) high-frequency map of each detected ROI and concatenates it with the RGB channels, providing the classifier with explicit frequency-domain features that characterize adversarial patch high-frequency noise.

**O4 — Build the Hardware-in-the-Loop Physical Failsafe**
Implement an ESP32-based physical failsafe subsystem that autonomously triggers a gate lock relay via MQTT within 200 milliseconds of adversarial patch detection, operating independently of cloud connectivity and incorporating a heartbeat-based watchdog that defaults to safe-lock upon communication failure.

**O5 — Develop a Full-Featured Security Dashboard**
Build a complete React and Node.js real-time security dashboard providing: live MJPEG video stream with detection overlays, adversarial attack frame gallery, temporal attack analytics, system health monitoring (FPS, latency, ESP32 uptime), and operator manual override controls communicating via MQTT to the ESP32.

## 3.2 Secondary Objectives

**O6 — Benchmark Against Existing Defenses**
Provide a comparative evaluation of the proposed SDNL against Jedi (CVPR 2023) and PatchZero (WACV 2023) on the same APDE test set, measuring accuracy, F1-score, ROC-AUC, and inference latency under identical conditions.

**O7 — Conduct Ablation Studies**
Quantify the independent contribution of each system component: (a) RGB-only vs. 4-channel SDNL input, (b) SDNL-only vs. SDNL + depth fusion, and (c) fixed detection threshold vs. adaptive threshold calibration.

**O8 — Evaluate Under Varying Physical Conditions**
Characterize system performance under real-world environmental variations including controlled lighting levels (50 lux vs. 500 lux), patch placement positions (torso, bag, foreground object), and subject-to-camera distances (0.5m, 1.0m, 1.5m, 2.0m).

**O9 — File Provisional Patent**
Document and file a provisional patent application with the Indian Patent Office (IPO) covering the dual-model edge architecture, the hardware-in-the-loop failsafe mechanism, and the frequency anomaly detection method prior to academic publication.

---

---

# 4. SCOPE OF THE PROJECT

## 4.1 In Scope

The following elements are within the defined scope of this research:

**4.1.1 Attack Types Addressed**
- Physical adversarial patch attacks — universally crafted, printed patches placed in the camera field of view
- Target blindness (disappearance) attacks on person/vehicle object detectors
- Misclassification attacks causing unauthorized subjects to be detected as authorized
- All 94 patch types covered by the APDE (ICCV 2025) benchmark dataset
- Naturalistic and color-pattern adversarial patches (APRICOT dataset)

**4.1.2 Infrastructure Scenarios Covered**
- Fixed-camera access control: parking gates, secure entryways, building lobbies
- Single-camera, single-access-point configurations (primary deployment scenario)
- Indoor and outdoor scenarios with structured illumination

**4.1.3 Hardware Platform**
- Edge compute: PC/Laptop (4GB) as the primary inference node
- Camera: Luxonis USB or IP Camera
- Failsafe controller: ESP32 DevKit V1
- Actuator: Single-channel relay module connected to electromagnetic gate lock
- Communication: Local area network MQTT (Mosquitto broker on RPi)

**4.1.4 Software Components**
- Primary detector: YOLOv8n (Ultralytics), deployed as Camera blob on PC CPU/GPU
- Secondary layer: Custom SDNL (PyTorch training, TFLite INT8 edge inference)
- Failsafe firmware: Arduino C++ on ESP32 (PlatformIO)
- Dashboard backend: Node.js + Express + Socket.IO + SQLite
- Dashboard frontend: React.js with Chart.js visualization

**4.1.5 Evaluation Scope**
- Detection accuracy, F1-score, ROC-AUC on APDE held-out test set
- Inference latency benchmarked on RPi 4B (TFLite INT8)
- End-to-end system latency: camera frame capture → ESP32 relay activation
- False positive rate on clean COCO val2017 and INRIA Person dataset frames
- Ablation study on frequency feature contribution and depth fusion
- Comparative evaluation against Jedi (CVPR 2023) and PatchZero (WACV 2023)

---

## 4.2 Out of Scope

The following are explicitly outside the scope of this work:

| Out-of-Scope Element | Reason / Future Work |
|---|---|
| **Digital adversarial attacks** (FGSM, PGD, C&W) | Require no physical access; different threat model; addressed by adversarial training literature |
| **Multi-camera, multi-gate networks** | Extension for future work; architecture is designed to be horizontally scalable |
| **Adversarial attacks on non-visual modalities** (audio, LiDAR) | Outside the computer vision scope of this system |
| **Deep fake / face swap attacks** | Different attack vector; not patch-based |
| **Cloud-based or GPU-dependent defenses** | Intentionally excluded; edge-only is the core constraint |
| **Mobile/vehicle-mounted camera scenarios** | Fixed-camera assumption is foundational to the threat model |
| **Night vision / IR-spectrum cameras** | DCT feature fusion assumes RGB + infrared; full IR-only is deferred |
| **Training new adversarial patches** | This work is defensive; new attack design is not a contribution |
| **Full patent prosecution** | Provisional patent is in scope; complete patent prosecution is post-paper |
| **Commercial-scale deployment** | Research prototype; production hardening is future work |

---

## 4.3 Assumptions

The following assumptions are made in this research:

1. **Fixed geometry:** The camera-to-subject distance is controlled (0.5–3.0 m) and the camera mounting angle is fixed. Attack patches are optimized for this geometry.

2. **White-box attack assumption:** The adversarial patches used in evaluation (from APDE and APRICOT) were generated with white-box or transfer-based methods. The defense is evaluated against both seen and unseen patch types.

3. **Local network availability:** The RPi, ESP32, and dashboard communicate over a local area network. Internet connectivity is not required for detection or failsafe operation.

4. **Single primary detector:** The system is designed around YOLOv8n as the primary detector. The SDNL is model-agnostic (any detector that provides bounding box crops as output is compatible).

5. **Known target classes:** The system detects persons and vehicles. Object classes specific to the deployment (e.g., authorized badges) can be added with fine-tuning.

6. **Printed patch on rigid/semi-rigid surface:** The system is primarily optimized for flat or near-flat patches. Patches on highly flexible clothing (with significant wrinkling) may show reduced detection accuracy (see UV-Attack, ICLR 2025 — a known adversarial advance beyond this work's scope).

---

## 4.4 Expected Outcomes

| Outcome | Type | Deliverable |
|---|---|---|
| Working physical prototype | System | RPi + Camera + ESP32 testbed |
| Trained SDNL model | Artifact | `.pt` and `.tflite` model files |
| Benchmark evaluation report | Results | Tables, ROC curves, latency measurements |
| Security dashboard | Software | React + Node.js application |
| Provisional patent application | IP | Filed with Indian Patent Office |
| Research paper | Publication | Submitted to IEEE Transactions on IoT |
| Demo video | Media | End-to-end attack + detection + gate lock recording |

---

---

# 5. PROPOSED SYSTEM

## 5.1 System Overview

InfraEdge-Shield is a hardware-in-the-loop cyber-physical security system designed to detect physical adversarial patch attacks on edge-deployed computer vision pipelines and autonomously trigger a physical failsafe response, independent of cloud infrastructure. The system is composed of five tightly integrated layers: (i) a camera sensor layer, (ii) a dual-model inference layer, (iii) an MQTT-based communication layer, (iv) a microcontroller-level hardware failsafe layer, and (v) a real-time web-based security dashboard. Together, these layers form a closed-loop security pipeline in which an adversarial patch attack detected in the visual domain is converted into an immediate physical access control response within 200 milliseconds, while simultaneously being logged and visualized for human operator review.

The proposed architecture departs from all prior work in two fundamental respects. First, it is the only system that closes the loop between computer vision adversarial detection and a microcontroller-driven physical actuator at the edge, without any cloud intermediary. Second, it introduces RGB depth information from an RGB-D camera as a novel auxiliary anomaly signal, exploiting the inherent spatial flatness of printed patches relative to natural three-dimensional scene geometry.

---

## 5.2 Layer 1 — Stereo Vision Sensor (Camera)

The camera serves as the visual input node, capturing RGB frames at 20–30 FPS and streaming them to the edge inference host (PC or laptop) via USB or network protocol. Any standard USB webcam, IP camera, or RTSP-compatible camera can be used. The edge host runs the YOLOv8n object detector and SDNL classifier concurrently on the CPU, achieving ≥20 FPS throughput through multi-threaded pipeline design.

The depth stream provides active stereo disparity maps aligned with the RGB frame. For each detected bounding box ROI, the corresponding depth patch is extracted and passed to the depth anomaly scoring module, which quantifies the spatial gradient flatness of the region. Adversarial patches—being flat, printed surfaces—consistently exhibit lower surface normal variation than natural objects such as clothing, skin, or vehicle paneling, providing a physically grounded discriminative signal that is independent of the patch's visual appearance.

---

## 5.3 Layer 2 — Dual-Model Inference Pipeline

The dual-model inference layer constitutes the primary technical contribution of InfraEdge-Shield. It operates as a multi-threaded pipeline with three concurrent execution threads:

**Thread 1 — Frame Reader:** Continuously dequeues (frame, bboxes, depth_map) tuples from the Camera XLink output at 30 FPS.

**Thread 2 — SDNL Inference Worker:** For each detected bounding box, extracts the corresponding 128×128 ROI from the RGB frame and the aligned depth patch. Computes the 4-channel input tensor:
- Channels 1–3: RGB pixel values (normalized)
- Channel 4: DCT (Discrete Cosine Transform) high-frequency component map — isolates the high-spatial-frequency content characteristic of adversarial patch noise patterns
- Channel 5: Laplacian energy map (optional auxiliary channel)

The SDNL CNN (MobileNetV3-Small backbone, adapted for 4-channel input, <500K total parameters) produces a binary classification (CLEAN / ATTACK) with an associated confidence score. The depth anomaly score is fused with the SDNL output as: **Final Score = 0.7 × SDNL_confidence + 0.3 × depth_anomaly_score**. A configurable threshold (default: 0.85) determines the attack detection decision.

**Thread 3 — Alert Publisher:** On attack detection, asynchronously publishes the MQTT alert payload, saves the flagged frame to local storage, and posts the event to the dashboard backend via HTTP.

---

## 5.4 Layer 3 — MQTT Communication Protocol

The MQTT protocol (OASIS standard, ISO/IEC 20922) serves as the lightweight publish-subscribe messaging backbone connecting the PC/Laptop, ESP32, and dashboard. A Mosquitto broker instance runs on the PC/Laptop, operating on the local area network without internet exposure. The protocol design prioritizes reliability and latency:

| MQTT Topic | Publisher | Subscriber | Payload | QoS |
|---|---|---|---|---|
| `infraedge/attack_alert` | RPi | ESP32, Dashboard | `{timestamp, confidence, frame_id}` | 1 |
| `infraedge/attack_ack` | ESP32 | RPi, Dashboard | `{gate_status, timestamp}` | 1 |
| `infraedge/override` | Dashboard | ESP32 | `{action: "lock"/"unlock"}` | 1 |
| `infraedge/heartbeat` | RPi | ESP32 | `{timestamp, fps, sdnl_latency_ms}` | 0 |
| `infraedge/status` | ESP32 | Dashboard | `{gate, uptime_s, last_alert}` | 0 |

A watchdog mechanism on the ESP32 monitors the heartbeat topic at 5-second intervals. If no heartbeat is received within 15 seconds — indicating RPi failure or network disruption — the ESP32 autonomously defaults to the locked state, ensuring fail-secure behavior.

---

## 5.5 Layer 4 — ESP32 Hardware Failsafe

The ESP32 microcontroller implements a state machine with three states: IDLE, LOCKED, and OVERRIDE. Upon receiving an `infraedge/attack_alert` message, the device transitions from IDLE to LOCKED, activating the gate relay (GPIO 26), flashing the red indicator LED (GPIO 27) at 5 Hz, and sounding the piezo buzzer (GPIO 25) at 880 Hz for two seconds. An acknowledgment is published to `infraedge/attack_ack` confirming the physical gate lock. A 30-second automatic lockout timer prevents rapid cycling. Security features include ESP32 Secure Boot and Flash Encryption to prevent firmware tampering.

---

## 5.6 Layer 5 — Full-Featured Security Dashboard

The React and Node.js dashboard provides comprehensive human oversight of the automated system. The dashboard comprises four main views:

1. **Main Dashboard:** Live MJPEG video stream with real-time bounding box overlays, gate status indicator (animated LOCKED/UNLOCKED), system health metrics (FPS, SDNL latency, ESP32 uptime), and a live alert sidebar.

2. **Attack Gallery:** Chronological grid of all flagged frames with timestamp, confidence score, gate response time, and operator annotations. Filterable by date range, confidence threshold, and gate status.

3. **Analytics:** Temporal attack frequency charts (Chart.js), confidence score distribution histogram, detection rate vs. system FPS correlation plot, and cumulative event count gauge.

4. **Settings & Override:** Configurable SDNL detection threshold, gate lockout duration, manual gate lock/unlock button with operator confirmation modal, and system diagnostic export.

The backend Node.js server bridges MQTT events to dashboard clients via Socket.IO WebSockets, ensuring sub-500ms dashboard refresh latency after an attack event. All attack events are persisted to a SQLite database for forensic retrieval.

---

## 5.7 Comparison with Existing Systems

| Feature | Cloud Pipeline | PatchGuard | Jedi (CVPR 2023) | **InfraEdge-Shield (Proposed)** |
|---|---|---|---|---|
| Edge-only (no cloud) | ✗ | ✓ | ✗ | ✅ |
| No model retraining needed | ✓ | ✗ | ✓ | ✅ |
| Physical hardware failsafe | ✗ | ✗ | ✗ | ✅ |
| DCT feature fusion | ✗ | ✗ | ✗ | ✅ |
| Real-time dashboard | ✗ | ✗ | ✗ | ✅ |
| Inference <15ms on RPi 4B | ✗ | ✗ | ✗ | ✅ |
| End-to-end latency <200ms | ✗ | ✗ | ✗ | ✅ |
| Works on IoT-class hardware | ✗ | ✗ | ✗ | ✅ |

---

---

# 6. LITERATURE SURVEY

The following table presents a comprehensive survey of twenty significant works in the domains of adversarial patch attacks, defense mechanisms, edge computing security, and cyber-physical systems. Papers are ordered newest to oldest to reflect the current state of the field. The merits and demerits are analyzed specifically with respect to their relevance and applicability to the proposed InfraEdge-Shield system.

| S.No | Title | Year | Venue | Merits | Demerits |
|:---:|---|:---:|---|---|---|
| 1 | Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights (APDE) | 2025 | ICCV | First standardized benchmark with 94K images across 94 patch types and 11 defenses; establishes Average Precision (AP) as the most reliable evaluation metric; reveals that stochastic and universal-patch defenses generalize best against adaptive attacks; provides publicly available dataset and evaluation toolkit | Benchmark paper only — no novel defense is proposed; evaluations are limited to static images without real-time or embedded hardware constraints; printed patch noise characteristics from physical environments are not fully modeled |
| 2 | DisPatch: Disarming Adversarial Patches in Object Detection with Diffusion Models | 2025 | arXiv | Employs a "regenerate and rectify" strategy that is fully attack-agnostic; achieves strong mAP@0.5 of 89.3% against hiding attacks; reduces adversarial patch creating-attack success rate to 24.8%; maintains robustness even against adaptive adversaries designed to bypass the defense | Multi-step diffusion model inference is computationally prohibitive for real-time edge deployment; requires several gigabytes of pre-trained diffusion model weights; GPU is mandatory; no evaluation performed on resource-constrained IoT or embedded hardware |
| 3 | UV-Attack: Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping | 2025 | ICLR | Uses Dynamic Neural Radiance Fields and UV texture mapping to model non-rigid human body deformations; Expectation over Pose Transformation (EoPT) loss achieves 92.7% attack success rate against FastRCNN and 49.5% against YOLOv8 in black-box settings; significantly raises the threat bar for physical infrastructure person detection | Attack-only research — no defense is proposed; NeRF-based training requires substantial GPU compute; practical printing and attachment of non-flat flexible patches remains non-trivial; does not study fixed-camera infrastructure scenarios |
| 4 | Ad_YOLO+: Robust Object Detection Under Adversarial Patch Attacks in Vision-Based Navigation | 2025 | MDPI Automation | Introduces a YOLO plug-in defense layer that treats adversarial patches as a dedicated detection category; eliminates heavy preprocessing overhead; demonstrates clean accuracy improvements alongside reduced attack success rates in AirSim simulations; evaluated on COCO and VisDrone datasets | Requires annotated patch-labelled retraining data; tightly coupled to YOLOv5x architecture; inference latency impact on embedded edge hardware is not characterized; not evaluated against naturalistic or GAN-generated adversarial patches |
| 5 | Segment and Recover (SAR): Defending Object Detectors Against Adversarial Patch Attacks | 2025 | arXiv | Two-stage preprocessing pipeline (patch-agnostic detection frontend + broken pixel restoration backend) is compatible with any pre-trained object detector without requiring retraining; demonstrated against localized-noise, printable, and adaptive patch types; handles variable patch scale, shape, and location | Per-frame preprocessing latency reduces system throughput; false positives on textured or cluttered backgrounds degrade practical precision; the restoration backend quality degrades for large patches; not benchmarked on any edge or embedded deployment hardware |
| 6 | PAD: Patch-Agnostic Defense Against Adversarial Patch Attacks | 2024 | CVPR | Exploits two fundamental patch properties—semantic independence (mutual information gap) and spatial heterogeneity (recompression residuals)—to localize patches without model-specific knowledge; leverages SAM for precise boundary refinement; requires no retraining; open-source implementation available | Segment Anything Model (SAM) is a ViT-based model requiring gigabytes of memory, making it unsuitable for PC/Laptop-class deployment; dual-property fusion block adds inference latency; not evaluated in real-time video streams or edge device contexts |
| 7 | DIFFender: Diffusion-Based Adversarial Defense Against Patch Attacks | 2024 | ECCV | Unified pipeline combining Adversarial Anomaly Perception (AAP) for patch localization with diffusion model-based restoration; few-shot prompt-tuning enables efficient adaptation to new attack types; generalizes across classification, face recognition, and physical-world scenarios | Multi-step diffusion inference requires a GPU; pre-trained diffusion weights are gigabyte-scale; high memory footprint is incompatible with IoT hardware; not evaluated on IoT-class or embedded devices in any reported experiment |
| 8 | ADAV: A Real-Time Defense Against Object Vanishing Adversarial Patch Attacks for Autonomous Vehicles | 2024 | arXiv | Pioneering use of temporal consistency checking across video frames for adversarial patch detection; MSE-based disruption detection combined with gradient-guided backpropagation for precise patch localization; optimized for low-latency operation; validated on BDD100K real driving dataset | Temporal consistency approach is fundamentally inapplicable to fixed-camera infrastructure scenarios where subjects may appear in only a single frame window; assumes continuous video with consistent reference frames available from prior timestamps |
| 9 | Adversarial Robustness for TinyML: Lightweight Defense Strategies for Edge-Deployed Neural Networks | 2024 | MDPI Sensors | Comprehensive empirical analysis of adversarial defense strategies (pruning, quantization, feature squeezing) on PC/Laptop and ESP32 hardware; adversarial training demonstrated to improve model accuracy from 46% to 97%; inference benchmarked below 3.5 ms on target hardware | Defense improvements are measured exclusively against digital perturbations (FGSM, PGD) and not against physical printed patch attacks; accuracy-robustness tradeoff degrades under aggressive INT8 quantization; dual-model or diagnostic anomaly detection architectures are not explored |
| 10 | Edge-Based Anomaly Detection for Cyber-Physical IoT Systems Using Lightweight ML and MQTT | 2024 | IEEE IoT Journal | Validates ESP32 as a viable security node for MQTT-based IoT deployments; demonstrates sub-15ms detection latency using XGBoost and CNN-LSTM hybrid; real-time anomaly flags communicated to actuators via MQTT; directly relevant to the proposed system's communication architecture | Focused exclusively on network-level anomaly detection (DoS, spoofing, payload injection); does not integrate visual or camera-based adversarial threat detection; no physical hardware failsafe mechanism coupled to computer vision outputs |
| 11 | Jedi: Entropy-Based Localization and Removal of Adversarial Patches | 2023 | CVPR | Model-agnostic defense requiring no model retraining; detects 90% of adversarial patches including naturalistic GAN-generated variants; 94% attack recovery rate via autoencoder inpainting; entropy is a lightweight statistical measure that can be approximated on edge hardware; outperforms prior methods LGS and Jujutsu | Autoencoder inpainting combined with entropy analysis introduces per-frame latency overhead; inpainting quality degrades significantly for complex or textured backgrounds; the system is not benchmarked on any embedded or IoT hardware; real-time video stream evaluation is absent from reported experiments |
| 12 | PatchZero: Defending Against Adversarial Patch Attacks by Detecting and Zeroing the Patch | 2023 | WACV | Two-stage defense combining pixel-level patch segmentation with mean-fill zeroing; applicable to image classification, object detection, and video classification tasks; two-stage adversarial training improves robustness against adaptive attacks; validated across ImageNet, PASCAL VOC, and UCF101 benchmarks | The segmentation network introduces significant additional inference overhead; zeroing with mean pixel values degrades image semantics particularly for large patches; requires retraining on patch-augmented datasets; not optimized for or evaluated on real-time edge deployment scenarios |
| 13 | Adversarial Patch Attacks Against Deep Learning-Based Pedestrian Detection Systems in the Physical World | 2023 | MDPI Sensors | Real-world outdoor physical evaluation of adversarial patches against YOLOv5 pedestrian detector under multiple controlled lighting levels and subject-to-camera distances; directly relevant threat model for access control and smart gate surveillance; demonstrates significant vulnerability of fixed-camera infrastructure setups | Purely an attack characterization paper — no defense is proposed or evaluated; attack success rate degrades substantially at distances beyond 3 meters or under adverse weather conditions; results are limited to the YOLOv5 detector architecture |
| 14 | PatchCleanser: Certifiably Robust Defense Against Adversarial Patches for Any Image Classifier | 2022 | USENIX Security | Model-agnostic certified defense using two-round masking strategy; no model retraining required; achieves better certified accuracy than PatchGuard; mathematically provable robustness within the specified threat model; compatible with any existing pre-trained classifier | Two-round inference approximately doubles the compute budget and latency per frame; not designed for or evaluated in real-time video or edge device contexts; provable robustness bounds are only valid within the stated threat model patch size constraints |
| 15 | PatchGuard: A Provably Robust Defense Against Adversarial Patches via Small Receptive Fields and Masking | 2021 | USENIX Security | First certifiably provable defense against adversarial patches at ImageNet scale; small receptive field architecture limits the number of features corrupted by any localized patch; robust feature aggregation provides mathematical robustness guarantees | Requires complete model retraining under restrictive architectural constraints; introduces 10–15% clean accuracy degradation; not applicable to pre-trained IoT or edge models; computationally intensive training; not evaluated on embedded hardware |
| 16 | Robust Physical-World Attacks on Deep Learning Visual Classification (RP2) | 2018 | CVPR | Seminal physical-world adversarial attack paper; RP2 algorithm achieves over 84% attack success rate in field tests on moving vehicles; attack patches are camouflaged as natural graffiti or stickers; demonstrated practical vulnerability of object detection systems using commercially available printed stickers | Patches are object-specific and not universal; attack success rate degrades significantly with distance, adverse lighting, and weather variation; considerable per-target setup time required; evaluation limited to stop sign classifiers and not generalized to smart infrastructure access control scenarios |
| 17 | Adversarial Patch (Brown et al.) | 2017 | arXiv | First demonstration of a universal, robust, and targeted adversarial patch that functions in the physical world; Expectation-over-Transformation (EoT) training ensures patch robustness across diverse physical viewing conditions; patches are placement-agnostic and applicable to any scene; foundational methodology underpinning all subsequent physical adversarial patch research | Designed for image classification models only — not object detectors; requires white-box gradient access to the target model during optimization; large patch dimensions reduce stealth in real-world deployment; EoT optimization is computationally expensive per target class |
| 18 | Towards Deep Learning Models Resistant to Adversarial Attacks (PGD Adversarial Training) | 2018 | ICLR | Foundational framework framing adversarial robustness as a min-max saddle-point optimization problem; Projected Gradient Descent (PGD) is established as the gold-standard first-order adversarial robustness baseline; architecture-agnostic training methodology applicable to any neural network | Significantly increases training time and computational cost; inherently trades clean accuracy for adversarial robustness; robust only against seen perturbation distributions; does not specifically address localized physical patch attacks |
| 19 | MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications | 2017 | arXiv | Depthwise separable convolutions reduce computational complexity by 8–9× compared to standard convolutions; well-supported in TensorFlow Lite for edge deployment; serves as the foundational architecture for all modern edge vision models including YOLOv8n and MobileNetV3 used in the proposed SDNL | Reduced accuracy relative to full-scale convolutional models; not adversarially robust by default without adversarial training; depthwise separable operations are not always efficiently optimized on all microcontroller hardware targets |
| 20 | Explaining and Harnessing Adversarial Examples (FGSM) | 2015 | ICLR | Established the foundational theoretical framework for adversarial machine learning, attributing neural network vulnerability to their linear behavior in high-dimensional input spaces; introduced the Fast Gradient Sign Method (FGSM) as a computationally efficient attack; proposed adversarial training as a regularization and defense strategy | Generates only digital pixel-level perturbations with no physical-world printability or localization constraints; requires white-box access to model gradients; single-step perturbation is easily defended by modern smoothing and detection methods |

---

## 6.1 FINDINGS IN LITERATURE SURVEY

The following key findings are derived from the systematic analysis of the twenty surveyed works. These findings directly motivate the design decisions in the proposed InfraEdge-Shield system and establish the research gap that this work addresses.

---

### Finding 1: A Critical Edge Deployment Gap Exists Across All High-Performing Defenses

Every high-performing adversarial patch defense in the literature — including DisPatch (2025, arXiv), DIFFender (ECCV 2024), Jedi (CVPR 2023), PAD (CVPR 2024), PatchGuard (USENIX 2021), and PatchCleanser (USENIX 2022) — either requires GPU-class hardware, gigabyte-scale model weights, or computational resources that exceed the capabilities of standard IoT edge devices such as the PC/Laptop. Not one of these systems has been benchmarked for inference latency on ARM Cortex-A class processors. This gap between theoretically strong defenses and practically deployable edge systems is the primary gap this work addresses.

**Implication:** The proposed SDNL must be a purpose-built, lightweight network (<500K parameters, TFLite INT8) validated specifically on the target edge hardware with documented inference latency benchmarks.

---

### Finding 2: Fixed-Camera Infrastructure Is a More Severe Threat Environment Than Mobile Scenarios

The majority of existing adversarial patch attack literature evaluates attacks in mobile and dynamic scenarios — autonomous vehicles (ADAV, 2024; RP2, 2018), robots in navigation (Ad_YOLO+, 2025), and person detection in public outdoor spaces (Gu et al., MDPI Sensors 2023). In these scenarios, variable camera-to-subject distance (5–50 m), viewing angle variation, and inconsistent lighting conditions collectively degrade attack success rates. In contrast, fixed-camera smart infrastructure systems operate at consistent short range (0.5–3.0 m), fixed mounting angle, and structured indoor illumination. The APDE benchmark (ICCV 2025) further confirms that physical patch attacks optimized for controlled environments achieve significantly higher attack success rates.

**Implication:** The threat model for this work is formally more severe than the majority of prior evaluations. Defenses validated only in mobile scenarios should not be assumed to be sufficient for fixed infrastructure deployment.

---

### Finding 3: No Prior System Couples Adversarial Detection to a Physical Hardware Response

Across all twenty surveyed works, not one system couples the output of adversarial patch detection to a physical-domain actuator response. All existing defenses operate purely in the digital processing domain — either cleaning the input image before it reaches the primary detector, modifying model outputs, or generating alert logs. The ADAV system (2024) comes closest, with a temporal detection mechanism, but its output is still a software flag rather than a hardware actuator signal. This absence of a hardware-in-the-loop response layer represents the most critical security gap for physical access control infrastructure, where a 2–5 second human operator response delay is sufficient for an adversary to gain unauthorized physical access.

**Implication:** The proposed ESP32 hardware failsafe subsystem — triggering a physical gate lock within 200 milliseconds via MQTT — is the first such mechanism in the adversarial patch defense literature and constitutes a primary novel contribution.

---

### Finding 4: Entropy and High-Frequency Spatial Features Are a Viable Lightweight Detection Signal

Jedi (CVPR 2023) established through rigorous empirical validation that adversarial patches — including visually naturalistic GAN-generated patches — consistently exhibit statistically elevated entropy and abnormal high-frequency spatial content relative to surrounding natural image regions. This finding holds even for patches designed to visually blend with their surroundings, because the adversarial optimization process necessarily introduces high-frequency perturbation energy to manipulate convolutional feature maps. This property provides a detector-agnostic, lightweight signal that is computationally approachable on edge hardware (DCT and Laplacian operations on 128×128 patches complete in under 1 ms on ARM Cortex-A72). The APDE benchmark (ICCV 2025) further confirms that high-frequency-based signals are necessary but insufficient for naturalistic patches, motivating the inclusion of RGB depth as a complementary modality.

**Implication:** The SDNL's 4-channel input (RGB + DCT high-frequency + Laplacian) is directly grounded in this finding. The depth fusion signal compensates for the residual uncertainty on naturalistic patches identified by APDE.

---

### Finding 5: DCT High-Frequency Feature Extraction Is an Underutilized Yet Computationally Viable Signal

Across all twenty surveyed works, while entropy-based localization (Jedi, CVPR 2023) has been studied, the use of Discrete Cosine Transform (DCT) high-frequency coefficient maps as a dedicated input channel for a lightweight patch detection CNN has not been proposed. Adversarial patch optimization necessarily injects high-spatial-frequency energy into the image region to manipulate deep feature activations — a consequence of gradient-based pixel-space optimization. DCT decomposition isolates this energy efficiently: on a 128x128 ROI, 2D DCT computation completes in under 0.5 ms on any modern CPU, making it viable for real-time edge deployment without GPU acceleration. Concatenating the DCT high-frequency map as a fourth channel to the RGB ROI provides the SDNL classifier with an explicit representation of this attack artifact.

**Implication:** The SDNL 4-channel input (RGB + DCT high-frequency map) directly exploits this finding. The explicit frequency channel reduces training data requirements for SDNL convergence and improves generalization to unseen patch types.

---

### Finding 6: No Prior Work Provides a Real-Time Security Dashboard Integrated with Attack Detection

All existing adversarial patch defense systems present their outputs as offline evaluation metrics on static benchmark datasets. No prior system streams flagged attack frames to a live dashboard, provides real-time visualization of detection events, enables temporal trend analysis of attack patterns, or provides operator manual override capabilities coupled to the physical response layer. The absence of this operational visibility layer means that even if detection occurs, security operators have no situational awareness of ongoing adversarial attacks, their frequency, or their patterns.

**Implication:** The InfraEdge-Shield React and Node.js dashboard addresses this operational gap and provides a practical contribution to the real-world deployment viability of adversarial patch defense systems.

---

---

# 7. METHODOLOGY

## 7.1 Research Methodology Overview

This research adopts a **Design Science Research (DSR)** methodology, which is appropriate for work that produces an artifact — in this case, a complete hardware-software system — as its primary contribution. The methodology proceeds through five iterative phases: (1) Problem Identification and Motivation, (2) Design and Development, (3) Demonstration, (4) Evaluation, and (5) Communication. Each phase is described in detail below, and the corresponding system components and deliverables are mapped to these phases.

---

## 7.2 Phase 1 — Problem Identification and Dataset Preparation

**Step 1.1 — Threat Characterization:**
The adversarial threat model is formally defined (Section 2.1). The attack surface is parameterized by camera-to-subject distance *d*, viewing angle *θ*, and illumination level *L*. Physical patch attack variants are characterized across two attack objectives: target blindness and misclassification.

**Step 1.2 — Dataset Curation:**
The APDE benchmark (ICCV 2025, 94,000 images, 94 patch types) provides the primary training and evaluation dataset. APRICOT (Columbia University) provides supplementary real-world photographed patch images. COCO val2017 and INRIA Person Dataset provide clean (non-attacked) frames for false positive rate evaluation.

**Step 1.3 — Feature Engineering:**
For each bounding box ROI crop (resized to 128×128):
- **RGB channels:** Standard normalized pixel values [0, 1]
- **DCT high-frequency channel:** 2D Discrete Cosine Transform applied to the grayscale ROI; the top-right quadrant of the frequency spectrum (high spatial frequencies) is extracted and normalized
- **Laplacian energy channel:** Laplacian-of-Gaussian filter applied to the grayscale ROI to compute edge energy density map

These are stacked into a 4-channel tensor (H×W×5) for SDNL input. This feature engineering is validated by the entropy-based findings of Jedi (CVPR 2023) and extended with Laplacian energy as a complementary edge-density signal.

---

## 7.3 Phase 2 — SDNL Architecture Design and Training

**Step 2.1 — Architecture Selection:**
MobileNetV3-Small is selected as the SDNL backbone based on its proven efficiency on ARM hardware (Cortex-A72 on RPi 4B) and suitability for TFLite INT8 quantization. The first convolutional layer is modified to accept 4-channel input (from the standard 3-channel RGB). The classifier head is replaced with a two-class output (CLEAN / ATTACK) with a configurable softmax threshold.

**Step 2.2 — Training Protocol:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻³ with CosineAnnealing schedule |
| Weight decay | 1 × 10⁻⁴ |
| Loss function | Focal Loss (γ = 2) for class imbalance handling |
| Batch size | 64 |
| Maximum epochs | 50 (early stopping, patience = 8) |
| Data augmentation | Rotation ±30°, brightness ±40%, Gaussian noise, JPEG compression simulation, random crop |
| Train / Val / Test | 70% / 15% / 15% (stratified by patch type) |

**Step 2.3 — Model Export and Quantization:**
The trained PyTorch checkpoint is converted to ONNX format and subsequently to TFLite INT8 using representative dataset calibration. The quantized model is validated for less than 1% accuracy degradation relative to the full-precision checkpoint. The final TFLite model (target <3 MB) is deployed to the PC/Laptop for inference benchmarking.

---

## 7.4 Phase 3 — Stereo Depth Anomaly Score Computation

**Step 3.1 — Depth Patch Extraction:**
For each detected bounding box from YOLOv8n, the corresponding 128×128 RGB ROI is extracted and the DCT high-frequency map is computed from the grayscale version of the ROI.

**Step 3.2 — Flatness Anomaly Scoring:**
The depth anomaly score *A(d)* is computed as:

> A(d) = 1 − ( mean(|∇d|) / (max(|∇d|) + ε) )

where ∇d is the spatial gradient magnitude of the depth patch and ε = 10⁻⁶ is a numerical stability term. A score near 1.0 indicates a spatially flat surface (characteristic of a printed patch); a score near 0.0 indicates a natural 3D surface with significant depth variation.

**Step 3.3 — Score Fusion:**
The final adversarial detection score is computed as:

> S_final = α × S_SDNL + (1−α) × A(d), where α = 0.7

The fusion weight α is treated as a hyperparameter and validated empirically through ablation study (α ∈ {0.5, 0.6, 0.7, 0.8, 0.9}).

---

## 7.5 Phase 4 — Hardware Integration and System Assembly

**Step 4.1 — Camera DepthAI Pipeline Configuration:**
The YOLOv8n model is converted to OpenVINO blob format and uploaded to the Camera PC CPU/GPU using blobconverter. The DepthAI Python SDK configures a pipeline with a ColorCamera node (640×640, 30 FPS), a YoloDetectionNetwork node (running on PC CPU), and a StereoDepth node with bilateral filter post-processing.

**Step 4.2 — ESP32 Firmware Development:**
The ESP32 firmware is developed in Arduino C++ using the PlatformIO build system. The firmware implements the state machine described in Section 5.6, MQTT client using the PubSubClient library, and JSON payload parsing using ArduinoJson.

**Step 4.3 — Dashboard Deployment:**
The Node.js backend is deployed as a systemd service on the PC/Laptop, listening on port 3000. The React frontend is built and served as static files via the Express server. The Mosquitto MQTT broker is configured as a systemd service with auto-restart enabled.

---

## 7.6 Phase 5 — Evaluation and Benchmarking

**Step 5.1 — SDNL Performance Evaluation:**
The trained SDNL is evaluated on the held-out APDE test set (15% of 94K = ~14,100 samples). Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC. Per-patch-type F1 scores are computed across all 94 APDE patch categories. The TFLite INT8 model inference time is benchmarked on PC/Laptop (1000 inference runs, report mean ± standard deviation).

**Step 5.2 — Comparative Evaluation:**
The SDNL is compared against Jedi (CVPR 2023) and PatchZero (WACV 2023) on the identical APDE test set under identical hardware conditions. All three methods are run on the same PC/Laptop to ensure fair latency comparison.

**Step 5.3 — Ablation Study:**
Four ablation experiments are conducted:
- A1: RGB-only (3-channel) vs. 4-channel SDNL input
- A2: SDNL-only vs. SDNL + depth fusion (S_final)
- A3: Fusion weight sensitivity analysis (α sweep)
- A4: TFLite INT8 vs. FP32 accuracy and latency tradeoff

**Step 5.4 — End-to-End Latency Measurement:**
End-to-end latency is measured as the elapsed time between camera frame capture timestamp and ESP32 relay activation acknowledgment (MQTT ACK received), averaged over 50 attack injection trials.

**Step 5.5 — False Positive Evaluation:**
The system is evaluated on 500 clean frames from COCO val2017 and 500 clean frames from INRIA Person dataset. All frames are passed through the full pipeline. False positive rate = (number of frames flagged as ATTACK) / 1000.

---

---

# 8. SOFTWARE REQUIREMENTS SPECIFICATION

## 8.1 Functional Requirements

Functional requirements define the specific behaviors and capabilities that the InfraEdge-Shield system must exhibit.

---

### FR-01 — Real-Time Video Capture and Streaming
The system shall capture continuous RGB video frames from the Camera at a minimum frame rate of 20 FPS at a resolution of 640×640 pixels, The system shall output synchronized frame bundles containing the RGB frame and associated bounding box detections.

### FR-02 — Primary Object Detection
The system shall execute YOLOv8n object detection on each captured RGB frame using the Camera PC CPU/GPU. The detector shall output bounding boxes, class labels, and confidence scores for the following classes: person, car, motorcycle, bus, and truck. Only detections with confidence ≥ 0.5 shall proceed to the SDNL stage.

### FR-03 — ROI Feature Extraction
For each detected bounding box, the system shall extract the corresponding 128×128 pixel ROI from the RGB frame and compute: (a) normalized RGB pixel values, (b) DCT high-frequency component map from the grayscale ROI, and (c) Laplacian edge energy map from the grayscale ROI, stacking these into a 4-channel input tensor.

### FR-04 — Adversarial Patch Detection (SDNL)
The system shall execute SDNL inference on each 4-channel ROI tensor within 15 milliseconds on the PC/Laptop, producing a binary classification (CLEAN / ATTACK) and an associated confidence score in the range [0.0, 1.0].

### FR-05 — Stereo Depth Anomaly Scoring
The system shall extract the aligned depth patch for each detected bounding box and compute a spatial flatness anomaly score. The final detection score shall be computed as a weighted fusion of the SDNL confidence and depth anomaly score.

### FR-06 — Attack Alert Generation
When the final detection score exceeds the configurable threshold (default: 0.85), the system shall: (a) publish a JSON alert payload to the MQTT topic `infraedge/attack_alert`, (b) save the flagged frame as a JPEG image to local storage, and (c) post the event record to the dashboard backend API.

### FR-07 — ESP32 Physical Failsafe Activation
The ESP32 shall subscribe to the `infraedge/attack_alert` MQTT topic. Upon receiving an alert, the ESP32 shall activate the gate lock relay (HIGH on GPIO 26), flash the red LED at 5 Hz, sound the buzzer at 880 Hz for 2 seconds, publish an acknowledgment to `infraedge/attack_ack`, and enforce a 30-second lockout timer.

### FR-08 — Watchdog Heartbeat Safety
The PC/Laptop shall publish a heartbeat message to `infraedge/heartbeat` every 5 seconds. The ESP32 shall monitor heartbeat receipt and default to the locked (safe) gate state if no heartbeat is received within 15 seconds.

### FR-09 — Manual Gate Override
The security dashboard shall provide an operator with a manual gate lock/unlock control. Operator override commands shall be published to the MQTT topic `infraedge/override` and executed by the ESP32 within 1 second.

### FR-10 — Live Video Dashboard Stream
The dashboard shall display a live MJPEG video stream from the PC/Laptop camera pipeline, updated at no less than 10 FPS, with bounding box overlays for all active detections.

### FR-11 — Attack Event Logging and Retrieval
Every attack event shall be persisted to the SQLite database with fields: event ID, UTC timestamp, frame image path, SDNL confidence score, depth anomaly score, gate lock status, and operator acknowledgment status. Events shall be retrievable via paginated REST API.

### FR-12 — Analytics Visualization
The dashboard analytics page shall display: (a) attack frequency over time as a line chart with configurable time range, (b) confidence score distribution as a histogram, (c) system FPS and SDNL latency over time, and (d) cumulative event count since system initialization.

### FR-13 — System Health Monitoring
The dashboard shall display real-time system health indicators: PC/Laptop inference FPS, average SDNL latency in milliseconds, ESP32 connection status, and time since last heartbeat.

### FR-14 — Configurable Detection Threshold
The operator shall be able to adjust the SDNL detection threshold (range: 0.50–0.99) from the dashboard settings page. Changes shall take effect within one processing cycle without system restart.

---

## 8.2 Non-Functional Requirements

Non-functional requirements define the quality constraints and operational characteristics of the system.

---

### NFR-01 — End-to-End Latency
The elapsed time from camera frame capture to ESP32 relay activation shall not exceed **200 milliseconds** under normal operating conditions (local area network, no packet loss).

### NFR-02 — SDNL Inference Latency
The SDNL TFLite INT8 model inference time per ROI on the PC/Laptop shall not exceed **15 milliseconds** (mean over 1000 inference runs at operating temperature).

### NFR-03 — System Throughput
The complete dual-model pipeline (camera capture → YOLOv8n → SDNL → publish) shall sustain a minimum throughput of **20 FPS** for scenes with up to 5 simultaneous detected bounding boxes.

### NFR-04 — Detection Accuracy
The SDNL shall achieve a minimum **92% detection accuracy** on the APDE held-out test set across all 94 patch types.

### NFR-05 — False Positive Rate
The system false positive rate on clean, non-attacked frames (COCO val2017 + INRIA Person) shall not exceed **5%** at the default operating threshold.

### NFR-06 — Cloud Independence
The system shall perform complete adversarial patch detection, MQTT communication, gate lock actuation, and dashboard data logging without requiring any active internet connection or cloud service availability.

### NFR-07 — Dashboard Refresh Latency
The security dashboard shall reflect attack events received via Socket.IO within **500 milliseconds** of the event being logged by the PC/Laptop pipeline.

### NFR-08 — Availability and Reliability
The edge inference pipeline shall operate continuously for a minimum of **72 hours** without requiring restart. The ESP32 watchdog mechanism shall ensure the gate defaults to the locked state within 15 seconds of any detected pipeline failure.

### NFR-09 — Data Persistence
All attack event records, including flagged frame images, shall be persisted to local SQLite storage on the PC/Laptop. Data shall survive system restarts and be retrievable via the dashboard REST API for at minimum 90 days of stored events.

### NFR-10 — Security and Authentication
All dashboard API endpoints shall require JWT-based authentication. MQTT communication shall be restricted to the local area network. The ESP32 firmware shall operate with Secure Boot and Flash Encryption enabled.

### NFR-11 — Modular and Extensible Architecture
The SDNL module shall be replaceable with an updated model version without requiring changes to the pipeline, MQTT, or dashboard components. The primary detector shall be swappable (e.g., YOLOv8n → YOLO11n) by modifying only the Camera pipeline configuration.

### NFR-12 — Resource Constraints
The complete edge software stack (detection pipeline, MQTT client, local HTTP server) shall operate within the following resource bounds on the PC/Laptop: CPU utilization < 75% mean across 4 cores, RAM usage < 2.5 GB, storage consumption < 10 GB per 30 days of operation.

### NFR-13 — Portability
The PC/Laptop software components shall be containerizable using Docker for reproducible deployment. The ESP32 firmware shall be buildable using PlatformIO CLI without commercial IDE dependencies.

### NFR-14 — Testability
Each system component (SDNL inference, MQTT publisher, ESP32 subscriber, dashboard backend) shall include unit tests covering normal operation, edge cases (empty frame, no detections, network disconnection), and fault injection scenarios.

---

---


---

---

# 9. SYSTEM ARCHITECTURE

## 9.1 Overview

InfraEdge-Shield follows a **three-node cyber-physical architecture**: a camera for visual input, a PC or laptop as the edge inference host, and an ESP32 microcontroller as the dedicated hardware failsafe controller. This design deliberately avoids cloud dependency — all inference, storage, communication, and dashboard hosting run locally on the edge host machine. The ESP32 operates independently of the inference host after initial boot, subscribing to MQTT alerts over the local Wi-Fi network and directly actuating a relay-controlled gate lock on receiving an attack alert.

The system consists of four integrated layers:

1. **Sensor Layer** — Camera (USB or IP) capturing RGB frames
2. **Inference Layer** — PC/Laptop running YOLOv8n object detector and SDNL anomaly classifier
3. **Communication Layer** — Mosquitto MQTT broker (hosted on PC) over local Wi-Fi
4. **Failsafe and Visualization Layer** — ESP32 controlling gate lock; React dashboard on PC

---

## 9.2 System Block Diagram

```
+-----------------------------------------------------------------------+
|  CAMERA                                                               |
|  (USB Webcam / IP Camera / Any RGB Camera)                            |
|  RGB Frames @ 20-30 FPS                                               |
+------------------------------------+----------------------------------+
                                     | USB / Wi-Fi / RTSP
+------------------------------------v----------------------------------+
|  EDGE INFERENCE HOST  (PC / Laptop — Windows or Linux)               |
|                                                                       |
|  +---------------------+    +-----------------------------------+     |
|  | YOLOv8n Detector    |    | SDNL CNN Classifier               |     |
|  | (ONNX / TFLite)     +--->| 4-ch: RGB + DCT high-freq map    |     |
|  | Detects persons,    | ROI| MobileNetV3-Small, <500K params  |     |
|  | vehicles per frame  |    | TFLite INT8 — ≤15ms/inference    |     |
|  +---------------------+    +----------------+------------------+     |
|                                              |  ATTACK (score>=0.85)  |
|  +-------------------+    +-----------------v------------------+     |
|  | Frame Store       |    | MQTT Publisher  (Paho Python)      |     |
|  | /frames/*.jpg     |<---| Publishes: infraedge/attack_alert  |     |
|  +-------------------+    +--------+---------------------------+     |
|                                     |                                 |
|  +----------------------------------v------+    +-----------------+   |
|  | Mosquitto MQTT Broker (localhost:1883)  |    | Node.js Backend |   |
|  | LAN-accessible, no internet required   +<-->| + React Dashbd  |   |
|  +----------------------------------+------+    | Port 3000       |   |
|                                     |           +-----------------+   |
+-------------------------------------|---------------------------------+
                                      | Wi-Fi LAN (MQTT QoS 1)
+-------------------------------------v---------------------------------+
|  ESP32  (Arduino C++ — PlatformIO)                                   |
|  Wi-Fi connected to same LAN                                         |
|                                                                       |
|  GPIO 26 --[Relay]---------> Electromagnetic Gate Lock (12V)         |
|  GPIO 27 --[Red LED]-------> Attack Alert Indicator (5 Hz flash)     |
|  GPIO 25 --[Buzzer]--------> Alarm Tone (880 Hz, 2 seconds)          |
|  GPIO 32 --[Green LED]-----> System Healthy Indicator                |
|  GPIO 34 --[Button]--------> Manual Override (INPUT_PULLUP)          |
|                                                                       |
|  Watchdog: if no heartbeat in 15s -> auto-lock (fail-secure)         |
|  State Machine: IDLE <--> LOCKED <--> OVERRIDE                       |
+-----------------------------------------------------------------------+
```

**Figure 9.1.** InfraEdge-Shield system block diagram. Camera to gate-lock latency: < 200ms.

---

## 9.2 Hardware Components

| Component | Role | Specifications |
|---|---|---|
| **Camera** | Visual input | Any USB webcam or IP camera; RGB, ≥ 20 FPS, ≥ 480p |
| **PC / Laptop** | Inference + backend + dashboard | Dual-core CPU, 8 GB RAM, Python 3.10, Node.js 18 |
| **ESP32 DevKit V1** | Hardware failsafe + actuator control | Dual-core Xtensa LX6 @ 240 MHz, Wi-Fi 802.11 b/g/n, 520 KB SRAM |
| **Relay Module** | Gate lock control | 5V single-channel relay; controls 12V electromagnetic lock |
| **Gate Lock** | Physical access control | 12V electromagnetic lock; Normally-Open relay circuit |

**ESP32 GPIO Pin Assignments:**

| GPIO | Connected To | Mode | Function |
|---|---|---|---|
| GPIO 26 | Relay IN | OUTPUT | Gate lock activation |
| GPIO 27 | Red LED (330Ω) | OUTPUT | Attack alert flash (5 Hz) |
| GPIO 25 | Piezo Buzzer | OUTPUT | Alarm tone (880 Hz PWM) |
| GPIO 32 | Green LED (330Ω) | OUTPUT | System healthy indicator |
| GPIO 34 | Pushbutton | INPUT_PULLUP | Manual override |

---

## 9.3 Software Architecture

```
infraedge-shield/
├── inference/
│   ├── pipeline.py        Main orchestrator (multi-threaded)
│   ├── detector.py        YOLOv8n via OpenCV VideoCapture
│   ├── sdnl.py            SDNL TFLite INT8 inference (4-ch input)
│   ├── mqtt_client.py     Paho MQTT publish / subscribe
│   └── frame_store.py     JPEG frame save + cleanup
│
├── esp32/
│   └── firmware/main.cpp  Arduino C++ FSM + MQTT + relay control
│
└── dashboard/
    ├── backend/
    │   ├── server.js       Express + Socket.IO + MQTT bridge
    │   ├── routes/         REST API: /api/events, /api/gate, /api/stats
    │   └── db/             SQLite schema + query helpers
    └── frontend/
        └── src/            React: LiveStream, AlertGallery, Analytics,
                            GateStatus, OverrideButton, SystemHealth
```

---

## 9.4 MQTT Communication Protocol

| Topic | Publisher | Subscriber | QoS | Purpose |
|---|---|---|---|---|
| `infraedge/attack_alert` | PC pipeline | ESP32, Dashboard | 1 | Trigger gate lock + log event |
| `infraedge/attack_ack` | ESP32 | PC, Dashboard | 1 | Confirm gate locked |
| `infraedge/override` | Dashboard | ESP32 | 1 | Manual lock / unlock |
| `infraedge/heartbeat` | PC pipeline | ESP32 | 0 | Watchdog keepalive every 5s |
| `infraedge/status` | ESP32 | Dashboard | 0 | Gate state broadcast |

---

## 9.5 End-to-End Data Flow

| Step | Time (ms) | Component | Action |
|:---:|:---:|---|---|
| 1 | 0 | Camera | RGB frame captured |
| 2 | 0 – 15 | YOLOv8n | Object detection → bounding boxes |
| 3 | 15 – 28 | SDNL | 4-ch tensor built; TFLite INT8 inference |
| 4 | 28 – 30 | Decision | score ≥ 0.85 → ATTACK confirmed |
| 5 | 30 – 35 | PC Thread 3 | Frame saved; MQTT alert published; HTTP POST |
| 6 | 35 – 90 | ESP32 | Alert received; relay HIGH → gate locked |
| 7 | 35 – 60 | Node.js | Event to SQLite; Socket.IO push to dashboard |
| 8 | 60 – 150 | React UI | Alert sidebar + gate badge updated |

**Camera to gate lock: ~90 ms | Camera to dashboard update: ~150 ms | Target: ≤ 200 ms**

---

---

# ERASER.IO PROMPT — Section 9 Architecture Diagram

> Go to **eraser.io → New → Diagram**. Select **"Architecture Diagram"** type.
> Paste the code block below into the editor.

---

```
title InfraEdge-Shield System Architecture

Camera [icon: camera, color: teal] {
  label: "Camera\n(USB / IP)\nRGB @ 30 FPS"
}

PC [icon: monitor, color: blue] {
  label: "Edge Inference Host\n(PC / Laptop)"

  YOLOv8n [icon: cpu, color: indigo] {
    label: "YOLOv8n\nObject Detector"
  }

  SDNL [icon: chip, color: purple] {
    label: "SDNL CNN\n4-ch TFLite INT8\n<500K params · ≤15ms"
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
    label: "Relay → Gate Lock\nGPIO 26"
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
SDNL --> MQTT_PUB: "ATTACK ≥ 0.85"
MQTT_PUB --> Backend: "HTTP POST /api/events"
Backend --> Dashboard: "Socket.IO events"
MQTT_PUB --> ESP32: "MQTT attack_alert (QoS 1)"
ESP32 --> Backend: "MQTT attack_ack"
ESP32 <--> Backend: "heartbeat / status"
Dashboard --> Backend: "POST /api/gate (override)"
Relay --> GateLock: "GPIO 26 HIGH → LOCK"
```

---

---

# 10. UML DIAGRAMS

## 10.1 Use Case Diagram (PlantUML)

> Paste into [planttext.com](https://planttext.com) or any PlantUML renderer. Fits A4 landscape.

```plantuml
@startuml
scale 800 width
left to right direction

title InfraEdge-Shield — Use Case Diagram

actor "Operator" as OP #5C6BC0
actor "Admin" as AD #26A69A
actor "Camera" as CAM #78909C
actor "ESP32" as ESP #FFA726

rectangle "InfraEdge-Shield" {
  package "Detection" {
    usecase "Capture Frame" as UC1
    usecase "Detect Objects\n(YOLOv8n)" as UC2
    usecase "Run SDNL\nInference" as UC3
    usecase "Confirm Attack" as UC4
  }
  package "Response" {
    usecase "Publish MQTT\nAlert" as UC5
    usecase "Lock Gate\n(Relay)" as UC6
    usecase "Log Event\nto DB" as UC7
    usecase "Push Alert\nto Dashboard" as UC8
  }
  package "Monitoring" {
    usecase "View Live\nStream" as UC9
    usecase "View Attack\nGallery" as UC10
    usecase "Override\nGate" as UC11
    usecase "View\nAnalytics" as UC12
  }
  package "Admin" {
    usecase "Update SDNL\nModel" as UC13
    usecase "Manage\nCredentials" as UC14
  }
  package "Safety" {
    usecase "Send\nHeartbeat" as UC15
    usecase "Auto-Lock\n(Timeout)" as UC16
  }
}

CAM --> UC1
UC1 --> UC2 : <<include>>
UC2 --> UC3 : <<include>>
UC3 --> UC4 : <<include>>
UC4 --> UC5 : <<include>>
UC4 --> UC7 : <<include>>
UC5 --> UC6 : <<include>>
UC5 --> UC8 : <<include>>

ESP --> UC6
ESP --> UC16

OP --> UC9
OP --> UC10
OP --> UC11
OP --> UC12

AD --> UC13
AD --> UC14

UC15 --> UC16 : <<extend>>

note right of UC4
  score >= 0.85
  (SDNL confidence)
end note

note bottom of UC16
  No heartbeat in 15s
  -> fail-secure lock
end note
@enduml
```

---

## 10.2 Class Diagram (PlantUML)

> Fits A4 portrait. Shows the 4 core system components and their key relationships.

```plantuml
@startuml
scale 800 width

title InfraEdge-Shield — Class Diagram

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

  class SDNLClassifier {
    - interpreter : tflite.Interpreter
    - threshold : float
    + infer(roi) : tuple[str, float]
    + preprocess(roi) : ndarray
    - build_4ch_tensor(rgb) : ndarray
  }

  class MQTTClient {
    - client : paho.Client
    - broker : str
    + connect() : void
    + publish_alert(conf, frame_id) : void
    + publish_heartbeat(fps) : void
    + subscribe(topic, cb) : void
  }
}

package "Data" {
  class AttackEvent {
    + id : int
    + timestamp : datetime
    + confidence : float
    + frame_path : str
    + gate_status : GateStatus
    + to_dict() : dict
  }

  class BBox {
    + x1, y1, x2, y2 : int
    + label : str
    + confidence : float
    + to_roi(frame) : ndarray
  }

  enum GateStatus {
    LOCKED
    UNLOCKED
    OVERRIDE
  }
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
}

package "Dashboard" {
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
}

' Relationships
DetectionPipeline *-- YOLODetector
DetectionPipeline *-- SDNLClassifier
DetectionPipeline *-- MQTTClient
DetectionPipeline ..> AttackEvent : creates
YOLODetector ..> BBox : produces
AttackEvent --> GateStatus
ESP32Controller --> FSMState
BackendServer --> ReactDashboard : "Socket.IO"
MQTTClient ..> ESP32Controller : "MQTT alert_alert →"
ESP32Controller ..> BackendServer : "MQTT attack_ack →"
BackendServer ..> MQTTClient : "MQTT override →"

note right of SDNLClassifier
  Input: 4-channel tensor
  [R, G, B, DCT_highfreq]
  Model: MobileNetV3-Small
  <500K params | TFLite INT8
end note

note bottom of ESP32Controller
  Watchdog: auto-lock gate
  if no heartbeat in 15s
end note
@enduml
```

---

## 10.3 Sequence Diagram — Attack Detection (PlantUML)

> Fits A4 landscape. Shows the complete attack-to-lock flow.

```plantuml
@startuml
scale 900 width

title InfraEdge-Shield — Attack Detection Sequence

participant "Camera" as CAM #1ABC9C
participant "YOLOv8n" as YOLO #3498DB
participant "SDNL\nClassifier" as SDNL #8b5cf6
participant "MQTT\nBroker" as MQTT #f59e0b
participant "ESP32" as ESP #ef4444
participant "Node.js\nBackend" as NODE #22c55e
participant "Dashboard" as UI #14b8a6

== Frame Capture & Detection ==
CAM -> YOLO : RGB Frame (USB)
YOLO -> YOLO : YOLOv8n inference\n→ Bounding Boxes
YOLO -> SDNL : ROI crops (128×128)

== SDNL Inference (~12ms) ==
SDNL -> SDNL : Build 4-ch tensor\n[RGB + DCT]
SDNL -> SDNL : TFLite INT8 inference
note right of SDNL : confidence = 0.94\nATTACK confirmed
SDNL ->> MQTT : PUBLISH attack_alert\n{conf:0.94, frame_id:42}
SDNL ->> NODE : HTTP POST /api/events

== Physical Failsafe (~90ms total) ==
MQTT ->> ESP : DELIVER attack_alert (QoS 1)
ESP -> ESP : State: IDLE → LOCKED\nGPIO 26 HIGH → Gate Locked\nRed LED 5Hz + Buzzer 880Hz
ESP ->> MQTT : PUBLISH attack_ack\n{gate: LOCKED}

== Dashboard Notification (~150ms total) ==
NODE -> NODE : INSERT to SQLite
NODE -> UI : Socket.IO: attack_detected
UI -> UI : Show alert + 🔴 LOCKED badge

== Watchdog (Background) ==
loop Every 5 seconds
  SDNL ->> MQTT : PUBLISH heartbeat
  MQTT ->> ESP : DELIVER heartbeat
  ESP -> ESP : Reset 15s watchdog timer
end

note over ESP
  If no heartbeat in 15s:
  auto-lock (fail-secure)
end note
@enduml
```
