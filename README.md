# InfraEdge-Shield — Real-Time Detection & Mitigation of Physical Adversarial Patch Attacks

Edge-native defense for camera-gated infrastructure: **YOLOv8n** (primary detector) +
**SDNL** 4-channel CNN (RGB + 2D DCT high-frequency map, MobileNetV3-Small, ONNX CPU) →
**ESP32** fail-secure relay lock over HMAC-signed MQTT → **React + Node** operator dashboard.

## Quickstart (demo, ~3 min)

```bash
./demo.sh                 # backend :3000 + frontend :5173 + pipeline, forced attack every 10th frame
./demo.sh --no-demo       # normal detection, no forced attacks
DEMO_EVERY=5 ./demo.sh    # forced attack every 5th frame
```

- Dashboard: `http://localhost:5173` — login `admin` / `password123`
- API health: `GET http://localhost:3000/api/health`
- Live frame: `GET http://localhost:3000/api/live-frame`

Prerequisites: `venv/` Python env (`requirements.txt`), Node + npm, optional Mosquitto
(`demo.sh` continues without it; backend POST path still works).

## Manual run

```bash
source venv/bin/activate
python inference/pipeline.py --dry-run --dummy --demo-attack-every 10 --max-frames 60
venv/bin/python -m unittest discover -s tests -v   # 18 smoke tests
venv/bin/python training/benchmark_sdnl.py --eval-only  # report → experiments/results/
cd dashboard/backend && npm start
cd dashboard/frontend && npm run dev
```

## Hard-negative retraining (texture FPs: screens, QR, text, fabric)

```bash
venv/bin/python training/synth_hard_negatives.py  # 360 synthetic CLEAN tensors → data/preprocessed/clean/synth_*.npy
venv/bin/python training/train_sdnl.py            # retrain (CrossEntropy, 20 epochs, CPU) → models/sdnl/
venv/bin/python training/benchmark_sdnl.py --eval-only  # verify recall held + FPR down
```

The clean set intentionally includes dense text, icon grids, QR-like codes, barcodes,
moiré screens, halftone print, and checkered/striped fabric — the everyday textures
that excite the DCT channel. Keep `models/sdnl/` backups before retraining (gitignored).

## Verified numbers (2026-10-12, purged COCO-clean + 360 synthetic texture negatives vs APRICOT-attack, 214 held-out samples)

| Accuracy | Precision | Recall | F1 | Threshold | FPR |
|---|---|---|---|---|---|
| 99.53% | 100.00% | 99.23% | 0.9961 | 0.30 operating (clean max 0.046, attack p5 0.699) | 0.00% |

External check: 300 unseen INRIA pedestrian photos → **1.33% FPR** @ 0.30 (top hit: window-grid facade).
`venv/bin/python training/fetch_inria_clean.py` reproduces the set (kept outside the training pool).

SDNL < 500K params, ONNX CPU edge-ready (≤ 15 ms/ROI target). Curves: `experiments/results/`.

## Security model

Zero-Trust HMAC-SHA256 on all `smartgate/*` MQTT topics (canonical form shared by
Python/JS/ESP32 firmware); HTTP event intake requires HMAC; gate override and alert
ack require JWT operator auth. ESP32 boots LOCKED, 15 s heartbeat watchdog, 30 s
auto-release. Decodable QR structured codes (badges/passes) are suppressed on the
ATTACK path (`qr_suppress`, `config.yaml`) since dense QR grids excite the DCT channel;
barcodes/textured signage can still false-positive — the durable fix is QR/texture
negatives in the clean set + retrain. See `docs/session_starter.md`, `docs/progress.md`, `.env.example`.
