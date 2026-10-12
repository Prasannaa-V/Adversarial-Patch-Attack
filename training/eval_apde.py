"""APDE cross-dataset evaluation (frozen v2 weights, fixed 0.30 threshold).

Reads the local APDE HF download (data/apde_hf/, see README download cmd),
runs the TEST split through the production decision path:
  YOLOv8n person proposals (conf 0.5, imgsz 416) -> SDNL 4-channel ONNX @0.30.

Reports:
  - recall + clean FPR with counts and Wilson 95% CIs
  - breakdowns by patch type / attack method / attack goal (worst first)
  - breakdown by patch area fraction of the 416x416 image
  - proposal-stage misses vs classifier-stage misses (GT-box scoring vs YOLO path)
  - overlap audit: APDE test source photos vs our COCO/INRIA/training pools

Constraints honored: v2 weights and eval_report.txt untouched, NO threshold
tuning on APDE test (0.30 fixed), all results reported including bad ones.

Outputs: experiments/results/apde_eval_report.txt (+ apde_eval_rows.csv)
Usage: venv/bin/python training/eval_apde.py [--limit N] [--stride K]
"""
import argparse
import csv
import glob
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from sdnl import build_4ch_tensor

THRESHOLD = 0.30
IMG = 416


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def parse_labels(text):
    persons, patches = [], []
    for line in (text or "").splitlines():
        f = line.split()
        if len(f) == 5 and f[0] in ("person", "patch"):
            box = list(map(float, f[1:]))
            (persons if f[0] == "person" else patches).append(box)
    return persons, patches


def crop128(bgr, box):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    return cv2.resize(bgr[y1:y2, x1:x2], (128, 128), interpolation=cv2.INTER_LINEAR)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Scorer:
    def __init__(self, onnx_path="models/sdnl/sdnl_v1.onnx"):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def score(self, roi_bgr):
        t = build_4ch_tensor(roi_bgr).transpose(2, 0, 1)[None, ...]
        logits = self.sess.run(None, {self.name: t})[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())


def our_coco_ids():
    ids = set()
    src = os.path.join("training", "download_real_clean_coco.py")
    try:
        txt = open(src).read()
        import re
        for m in re.finditer(r"000000(\d+)\.jpg", txt):
            ids.add(int(m.group(1)))
        for m in re.finditer(r"\b(\d{6})\b", txt):
            v = int(m.group(1))
            if 100 <= v <= 600000:
                ids.add(v)
    except OSError:
        pass
    return ids


def main():
    ap = argparse.ArgumentParser(description="APDE cross-dataset eval (frozen weights)")
    ap.add_argument("--limit", type=int, default=0, help="max test rows (0 = all)")
    ap.add_argument("--stride", type=int, default=1, help="use every Kth test row")
    ap.add_argument("--report", default="experiments/results/apde_eval_report.txt")
    args = ap.parse_args()

    from datasets import load_dataset
    from ultralytics import YOLO

    test_files = sorted(glob.glob(os.path.join("data", "apde_hf", "data", "test", "*.parquet")))
    if not test_files:
        raise SystemExit("APDE test parquet not found. Run the hf download first (see README).")
    ds = load_dataset("parquet", data_files={"test": test_files})["test"]
    if args.stride > 1:
        ds = ds.filter(lambda _, i: i % args.stride == 0, with_indices=True)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[apde] test rows: {len(ds)} (stride={args.stride}, limit={args.limit or 'all'})")

    yolo = YOLO("yolov8n.pt")
    scorer = Scorer()
    t0 = time.time()

    rows = []
    for r in ds:
        img = np.array(r["image"].convert("RGB"))  # 416x416 RGB
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        persons, patches = parse_labels(r.get("labels", ""))
        goal = r.get("attack_goal", "?")
        area_frac = (sum((b[2] - b[0]) * (b[3] - b[1]) for b in patches) / (IMG * IMG)
                     if patches else 0.0)
        # --- classifier-only path (GT boxes) ---
        if goal == "appearing" or not persons:
            gt_boxes, gt_kind = patches, "patch"
        else:
            gt_boxes, gt_kind = persons, "person"
        gt_scores = [scorer.score(crop128(bgr, b)) for b in gt_boxes] if gt_boxes else []
        gt_max = max(gt_scores) if gt_scores else 0.0
        # --- end-to-end path (YOLO proposals, production settings) ---
        res = yolo(bgr, imgsz=416, conf=0.5, verbose=False)[0]
        yolo_person = []
        for box in res.boxes:
            if int(box.cls[0]) < len(yolo.names) and yolo.names[int(box.cls[0])] == "person":
                yolo_person.append(list(map(float, box.xyxy[0])))
        matched = [b for b in yolo_person
                   if any(iou(b, g) >= 0.5 for g in persons)] if persons else []
        yolo_scores = [scorer.score(crop128(bgr, b)) for b in matched]
        yolo_max = max(yolo_scores) if yolo_scores else 0.0
        rows.append({
            "id": str(r.get("id", "")), "patch_id": str(r.get("patch_id", "?")),
            "method": str(r.get("method", "?")), "detector": str(r.get("detector", "?")),
            "goal": goal, "source": str(r.get("source_dataset", "?")),
            "source_id": str(r.get("source_id", "")),
            "area_frac": area_frac, "gt_kind": gt_kind, "gt_max": gt_max,
            "n_yolo": len(yolo_person), "n_matched": len(matched),
            "yolo_max": yolo_max,
            "gt_hit": gt_max >= THRESHOLD, "e2e_hit": yolo_max >= THRESHOLD,
        })
    dt = time.time() - t0
    print(f"[apde] scored {len(rows)} rows in {dt:.0f}s")

    # --- overlap audit vs our pools ---
    our_ids = our_coco_ids()
    apde_coco_ids = set()
    for r in rows:
        if r["source"] == "coco" and r["source_id"].isdigit():
            apde_coco_ids.add(int(r["source_id"]))
    overlap = sorted(our_ids & apde_coco_ids)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report.replace(".txt", "_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    L = []
    A = L.append
    A("InfraEdge-Shield SDNL — APDE cross-dataset evaluation")
    A("=" * 64)
    A(f"weights: frozen v2 (models/sdnl/sdnl_v1.onnx, UNCHANGED) | threshold: {THRESHOLD} FIXED (no tuning)")
    A(f"rows: {len(rows)} (test split, stride {args.stride}) | runtime: {dt:.0f}s")
    A("")
    hiding = [r for r in rows if r["goal"] == "hiding"]
    appearing = [r for r in rows if r["goal"] == "appearing"]
    other = [r for r in rows if r["goal"] not in ("hiding", "appearing")]
    A(f"attack goals present: {sorted(set(r['goal'] for r in rows))}")
    for name, grp in (("hiding", hiding), ("appearing", appearing), ("other", other)):
        if not grp:
            continue
        k = sum(r["gt_hit"] for r in grp)
        lo, hi = wilson(k, len(grp))
        A(f"[{name}] classifier recall: {k}/{len(grp)} = {k/len(grp)*100:.2f}% (95% CI {lo*100:.2f}-{hi*100:.2f})")
        if name == "hiding":
            e = sum(r["e2e_hit"] for r in grp)
            lo2, hi2 = wilson(e, len(grp))
            prop_miss = sum(1 for r in grp if r["n_matched"] == 0)
            clf_miss = sum(1 for r in grp if r["n_matched"] > 0 and not r["e2e_hit"])
            A(f"[{name}] end-to-end recall (YOLO+SDNL): {e}/{len(grp)} = {e/len(grp)*100:.2f}% (95% CI {lo2*100:.2f}-{hi2*100:.2f})")
            A(f"[{name}] miss split: proposal-stage (no matched person box) {prop_miss} | classifier-stage {clf_miss}")
        if name == "appearing":
            induced = sum(1 for r in grp if r["n_yolo"] > 0)
            A(f"[{name}] images where YOLO still found a person: {induced}/{len(grp)}")

    def breakdown(key, title):
        A("")
        A(f"--- by {title} (worst first) ---")
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        table = []
        for g, grp in groups.items():
            k = sum(x["gt_hit"] for x in grp)
            table.append((k / len(grp), g, k, len(grp)))
        for rate, g, k, n in sorted(table):
            lo, hi = wilson(k, n)
            A(f"  {g}: {k}/{n} = {rate*100:.2f}% (95% CI {lo*100:.2f}-{hi*100:.2f})")

    breakdown("patch_id", "patch type")
    breakdown("method", "attack method")
    breakdown("detector", "victim detector")

    A("")
    A("--- by patch area fraction of image ---")
    for name, lo_, hi_ in (("<1%", 0.0, 0.01), ("1-4%", 0.01, 0.04), (">4%", 0.04, 2.0)):
        grp = [r for r in rows if lo_ <= r["area_frac"] < hi_]
        if not grp:
            continue
        k = sum(r["gt_hit"] for r in grp)
        lo, hi = wilson(k, len(grp))
        A(f"  {name}: {k}/{len(grp)} = {k/len(grp)*100:.2f}% (95% CI {lo*100:.2f}-{hi*100:.2f})")

    A("")
    A("--- overlap audit (APDE test sources vs our pools) ---")
    A(f"our COCO val IDs in training pool: {len(our_ids)} | APDE-test COCO sources: {len(apde_coco_ids)}")
    A(f"overlapping COCO image IDs: {len(overlap)} {overlap[:20]}{'...' if len(overlap) > 20 else ''}")
    A("note: overlap means a familiar BACKGROUND only — no APDE patch type was ever trained on; "
      "recall impact, if any, is reported as-is, not corrected.")
    A("INRIA: our ext_inria mirror split is unknown, so INRIA-photo overlap cannot be excluded; "
      "APRICOT/synth pools cannot overlap APDE (INRIA+COCO sources only).")

    # clean FPR on APDE clean splits (downloaded alongside test)
    for split, label in (("positive", "clean-positive"), ("negative", "clean-negative")):
        files = sorted(glob.glob(os.path.join("data", "apde_hf", "clean", f"{split}*.parquet")))
        if not files:
            A(f"[{label}] clean parquet not downloaded — skipped");
            continue
        cds = load_dataset("parquet", data_files={split: files})[split]
        fp = n = 0
        for r in cds:
            bgr = cv2.cvtColor(np.array(r["image"].convert("RGB")), cv2.COLOR_RGB2BGR)
            persons, _ = parse_labels(r.get("labels", ""))
            boxes = persons if persons else [[62, 62, 354, 354]]
            s = max(scorer.score(crop128(bgr, b)) for b in boxes)
            n += 1
            fp += s >= THRESHOLD
        lo, hi = wilson(fp, n)
        A(f"[{label}] clean FPR: {fp}/{n} = {fp/n*100:.2f}% (95% CI {lo*100:.2f}-{hi*100:.2f})")

    A("")
    A("caveats: hiding patches here are SMALL (20% of person long side); expect dilution-driven "
      "low recall vs our tight-crop training. Clean sources are shared across APDE train/test "
      "patch groups (not source-disjoint) per the dataset card.")
    open(args.report, "w").write("\n".join(L) + "\n")
    print(f"[apde] report -> {args.report}")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
