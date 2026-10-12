# Demo runbook (5 minutes, minimal risk)

## One-time prep
- Print 2–3 patches from `data/crops/attack/` (~15–20 cm, matte paper).
- Record `demo_backup.mp4` with your phone: good light, each patch filling
  ≥ 1/4 of frame for 5 s + 10 s clean room. Fallback if live fails:
  `venv/bin/python inference/pipeline.py --video demo_backup.mp4 --threshold 0.60`
- Digital-only fallback (no print): `data/custom_test/demo_attack.png`
  (verified: pipeline ALERT conf 0.87 at threshold 0.60).

## Live script
```bash
./demo.sh  # backend :3000 + frontend :5173
venv/bin/python inference/pipeline.py --camera-url http://<phone>:8080/video \
  --threshold 0.60 --detect-every 2   # login admin / password123
```
1. Clean room → CLEAN, gate UNLOCKED (confs hover low).
2. Textured non-threat (mouse pad / screen) → stays CLEAN at 0.60.
3. Printed patch held close, filling frame → red ATTACK banner ~90%+, LOCKED.
4. Acknowledge the incident in the gallery → pending count drops.

## Safety rules (all measured, see docs/progress.md)
- Threshold **0.60 for the room**, not the 0.30 benchmark default: your
  environment's textures sit at 55–58%.
- Patch must cover ~1/3+ of the torso box; lap-level or distant patches miss
  (documented dilution limit). Operating range 0.5–3 m — say it out loud.
- One continuous hold = one incident (15 s alert cooldown); no spam on fumbles.
- Stable demo files: `demo_attack.png`, `image copy 13/14` (fire); avoid
  `image copy 12` (verdict flips 5–64% across runs — YOLO box jitter, known).
- If asked about misses: "patches under ~10% of the detection box are a
  measured 8%-recall blind spot, listed limitation with retrain path."
