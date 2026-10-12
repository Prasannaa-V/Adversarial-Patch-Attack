# Dissertation notes: scale generalization (all numbers committed in repo)

## Setup
SDNL (MobileNetV3-Small, RGB + 2D DCT, <500K params) trained on tight 128px
crops: clean COCO + 360 synthetic textures vs APRICOT patch crops. Operating
threshold 0.30 (clean max 0.046 vs attack p5 0.699 on the test split).

## Headline (same-distribution, n=214 held-out)
99.53% accuracy, 100% precision (95% CI 97.1–100), 99.23% recall, 0.00% FPR.
`experiments/results/eval_report.txt` (+ PR/ROC/confusion plots).

## Scale analysis (the honest core)
- Tight crops: 98.85% recall (873 APRICOT crops).
- In-scene composites (600, patches at 10–45% of frame): **8.2% recall**.
  Same weights, same threshold — dilution, not texture, is the failure mode.
- Live photo (person box, lap-level patch): whole-box 0.006 vs bottom-half
  **0.897** — led to person-region proposals (`person_regions` in
  `inference/sdnl.py`): whole + top/bottom halves, max wins.
- Box-geometry sensitivity: identical photo scores 5–64% across runs as YOLO
  boxes jitter; union-max over proposals + 15 s alert cooldown contain it.

## Negative results (report as rigor, not failure)
- Composite retraining attempted twice, both collapsed to majority-prior
  (v3: 22.6% FPR; v4: ROC-AUC 0.49) and were rejected; v2 weights kept live.
  Mechanism: pasted-patch backgrounds contradict clean labels, so the
  classifier learns the prior instead of the patch. Fixes shipped anyway:
  class weights, strict-best + patience early stopping, `--extra-attack` pool.
- Unseen gradient-noise patch family (LaVAN, since removed): 2.5–10%
  in-scene recall despite 16–18/20 classifier fooling rate — texture-family
  brittleness, future work (MIL head: per-location scores + max-pool).
- QR suppression audit: `detect()`-only fallback fired on 8.9% of real
  attacks (would flip 8.8% to CLEAN) → suppression restricted to decodable
  payloads (0/873 flips), residual glare-QR FPs contained by threshold.

## External probes
- 300 unseen INRIA pedestrian photos: 1.33% FPR (worst: window-grid facade).
- COCO clean: 0.50% FPR (dense foliage nearest the line at 0.577).
