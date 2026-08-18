# 🔬 FEASIBILITY ANALYSIS
## InfraEdge-Shield — Technical, Hardware & Publication Feasibility

---

## 1. Technical Feasibility

### 1.1 Dual-Model Pipeline on Edge Hardware

| Challenge | Risk | Mitigation |
|---|---|---|
| Running YOLOv8n + SDNL concurrently on RPi | Medium | OAK-D MyriadX NPU offloads YOLOv8n; RPi CPU focuses on SDNL only |
| SDNL inference <15ms on RPi 4B | Medium | TFLite INT8, MobileNetV3-Small (<500K params), 128×128 input |
| 20+ FPS end-to-end | Medium-High | OAK-D onboard detection pipeline; multi-threading with queue |
| High-frequency DCT computation per frame | Low | Scipy/OpenCV DCT on 128×128 ROI is <1ms on RPi 4B |
| MQTT latency RPi → ESP32 | Low | LAN MQTT typically <5ms; well within 200ms budget |

**Verdict: ✅ TECHNICALLY FEASIBLE** — validated by Paper #9 (Sensors 2024) which showed <3.5ms inference on RPi/ESP32 for comparable models, and Paper #10 (IEEE IoT 2024) showing <15ms MQTT-based edge security pipelines.

### 1.2 SDNL Detection of Physical Patches

| Challenge | Risk | Mitigation |
|---|---|---|
| High-freq signal sufficient for detection | Low | Jedi (CVPR 2023) proves entropy/high-freq works (90% detection rate) |
| Generalization across 94 patch types | Medium | APDE 94K dataset covers full diversity; augmentation adds robustness |
| False positives on natural textures | Medium | Laplacian+DCT multi-signal; threshold calibration; depth fusion |
| Printed patch color shift detection | Low | RGB channels + frequency maps together capture both |

**Verdict: ✅ TECHNICALLY FEASIBLE** — grounded in Jedi (CVPR 2023) which is the closest precedent; our approach extends it for edge hardware.

### 1.3 OAK-D Stereo Depth as Auxiliary Signal

| Challenge | Risk | Mitigation |
|---|---|---|
| Depth accuracy for small patch regions | Medium | OAK-D has 7.5cm minimum depth; patch at 0.5-2m distance is well in range |
| Depth noise in indoor environments | Medium | Use median-filtered depth; outlier rejection |
| Novel contribution (no prior work) | Low (risk: reviewers may not accept) | Frame as "extension" / supplementary material; validate empirically |

**Verdict: ✅ FEASIBLE + NOVEL** — adds a genuinely new dimension to physical patch detection.

---

## 2. Hardware Feasibility

### 2.1 Bill of Materials (Estimated)

| Component | Model | Est. Cost (INR) | Est. Cost (USD) |
|---|---|---|---|
| Edge Compute | Raspberry Pi 4B (4GB) | ₹5,500 | ~$65 |
| Stereo Camera | Luxonis OAK-D Lite | ₹12,000 | ~$149 |
| Failsafe MCU | ESP32 DevKit V1 | ₹400 | ~$5 |
| Relay Module | 5V 1-Channel Relay | ₹150 | ~$2 |
| LED + Buzzer kit | Assorted | ₹200 | ~$3 |
| Gate Lock (demo) | 12V Electromagnetic lock | ₹800 | ~$10 |
| Power Supply | 5V/3A for RPi, 12V for lock | ₹500 | ~$6 |
| MicroSD Card | 64GB Class 10 | ₹600 | ~$8 |
| Jumper wires, breadboard | — | ₹200 | ~$3 |
| **TOTAL** | | **~₹20,350** | **~$251** |

> **Alternative camera if OAK-D unavailable:** Intel RealSense D435i (~₹18,000 / ~$220) — similar stereo depth capabilities, DepthAI-compatible via OpenCV wrappers.

**Verdict: ✅ LOW-COST AND ACCESSIBLE** — total hardware cost <₹21,000 (~$251).

### 2.2 Compute Requirements for Training (Dev Machine)

| Component | Minimum | Recommended |
|---|---|---|
| GPU | GTX 1060 6GB | RTX 3060+ 12GB |
| RAM | 16 GB | 32 GB |
| Storage | 150 GB free | 500 GB SSD |
| Python | 3.9+ | 3.11 |
| CUDA | 11.8 | 12.1 |

**If no local GPU:** Use Google Colab Pro ($10/month) or Kaggle notebooks (free T4 GPU).

---

## 3. Dataset Feasibility

| Dataset | Size | Availability | Suitability |
|---|---|---|---|
| **APDE** (ICCV 2025) | 94K images, 94 patch types | Public: github.com/Gandolfczjh/APDE | ✅ Best — covers all attack variants, newest benchmark |
| **APRICOT** | ~2,500 physical patch photos | Public: apricot.cs.columbia.edu | ✅ Real printed patches in real environments |
| **INRIA Person** | 2,479 persons | Public: pascal.inrialpes.fr | ✅ Clean person frames for false-positive testing |
| **COCO val** | 5,000 images | Public: cocodataset.org | ✅ Clean scene frames for FPR evaluation |

**Total training data estimate:**
- APDE: ~65K train crops (70% of 94K)
- APRICOT: ~1,700 training samples
- Clean: ~5,000 COCO frames + 1,200 INRIA clean crops

**Verdict: ✅ SUFFICIENT DATA — no self-collection needed.**

---

## 4. Publication Feasibility

### 4.1 Gap Analysis vs. Literature

| What exists | What this project adds | Gap clearly identified? |
|---|---|---|
| Defenses are cloud/GPU dependent | Edge-only, <200ms response | ✅ Yes |
| No defense couples to hardware failsafe | ESP32 → relay → gate lock | ✅ Yes (strongest gap) |
| No prior work uses stereo depth as auxiliary signal | OAK-D depth fusion | ✅ Yes (novel extension) |
| No dashboard with live stream + hardware control | Full-featured React dashboard | ✅ Yes |
| Defenses not benchmarked on embedded hardware | Full RPi 4B latency benchmarks | ✅ Yes |

**Verdict: ✅ HIGH NOVELTY — 3–5 clearly novel contributions.**

### 4.2 Target Journals/Conferences

| Venue | Type | Impact Factor | Review Time | Fit |
|---|---|---|---|---|
| **IEEE Transactions on IoT** | Journal | 8.2 | 3–6 months | ✅ PRIMARY |
| **IEEE Internet of Things Magazine** | Magazine | 11.8 | 2–4 months | ✅ Alternative |
| **Computers & Security** | Journal | 5.6 | 3–5 months | ✅ Good fit |
| **MDPI Sensors** | Journal (OA) | 3.9 | 1–2 months | ✅ Fast track |
| **USENIX Security 2026** | Conference | Top-tier | ~10 months | ⚠️ Ambitious |
| **IEEE EDGE 2026** | Conference | Mid-tier | ~6 months | ✅ Excellent fit |

### 4.3 Rejection Risk Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| SDNL accuracy insufficient (<90%) | Low | Pre-validated by Jedi (90%) — we extend rather than start from scratch |
| Paper reviews say "no real-world eval" | Medium | Physical testbed + printed patches + photos/video mandatory |
| "Not novel enough" | Low | 3+ novel contributions (hardware failsafe + depth fusion + edge deployment) |
| "Compare with more baselines" | Medium | Implement Jedi + PatchZero as baseline comparisons in experiments |

---

## 5. Patent Feasibility

### 5.1 Patentability Checklist (Indian Patent Act 1970)

| Criterion | Assessment |
|---|---|
| **Novelty** | ✅ No existing patent combines: visual adversarial detection + edge MCU hardware failsafe + stereo depth |
| **Inventive Step** | ✅ Non-obvious to a person skilled in the art (security engineer would not trivially combine these) |
| **Industrial Application** | ✅ Direct application in parking gates, entryways, smart city cameras |
| **Not excluded** | ✅ Not a pure algorithm (has physical hardware component — satisfies Section 3(k) exception) |

### 5.2 Patent Filing Strategy

```
Step 1: File Provisional Patent Application (PPA)
        - Indian Patent Office (IPO): ~₹4,000 fee (individual)
        - Establishes priority date
        - Valid for 12 months

Step 2: Within 12 months of PPA:
        - File Complete Specification (full patent)
        - Consider PCT (Patent Cooperation Treaty) for international filing

Step 3: After filing PPA:
        - Safe to submit academic paper for publication
        - Include "Patent Pending" notice in paper

Estimated Cost: ₹4,000–₹15,000 (Indian filing, individual/startup)
Patent Attorney: Optional for PPA, recommended for complete spec
```

---

## 6. Timeline Feasibility Summary

| Phase | Weeks | Feasibility |
|---|---|---|
| Setup + Dataset | 2 | ✅ Straightforward |
| SDNL Training | 3 | ✅ With GPU (Colab) |
| Integration (edge) | 3 | ✅ Standard Python |
| ESP32 Firmware | 1 | ✅ Simple C++ MQTT |
| Dashboard | 3 | ✅ Standard web dev |
| Benchmarking | 3 | ✅ Well-defined experiments |
| Patent + Paper | 4 | ✅ Guided by this document |
| **Total** | **~20 weeks** | ✅ REALISTIC |

**Verdict: ✅ OVERALL FEASIBILITY = HIGH**
The project is well-scoped, hardware is accessible, datasets are available, novelty is clear, and publication path is defined.
