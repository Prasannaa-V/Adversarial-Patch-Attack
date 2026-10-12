"""Composite in-scene ATTACK samples: paste real APRICOT patch crops into real
COCO scenes at random scales/positions so training sees the live distribution
(full person-box ROIs with background), not just tight patch crops.

Writes data/crops/comp_attack/*.jpg + data/preprocessed/attack/comp_*.npy.
Deterministic (seed=11). Re-runnable: clears old comp_* first.
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from sdnl import build_4ch_tensor

SEED = 11
N_TARGET = 600


def main():
    rng = np.random.default_rng(SEED)
    patches = sorted(glob.glob(os.path.join("data", "crops", "attack", "*.jpg")))
    big_scenes = sorted(glob.glob(os.path.join("data", "coco_clean", "images", "*.jpg")))
    small_scenes = sorted(glob.glob(os.path.join("data", "crops", "clean", "coco_*.jpg")))
    assert patches, "no APRICOT patch crops found"
    assert big_scenes or small_scenes, "no COCO scene bases found"

    crops_dir = os.path.join("data", "crops", "comp_attack")
    prep_attack = os.path.join("data", "preprocessed", "attack")
    os.makedirs(crops_dir, exist_ok=True)
    for f in glob.glob(os.path.join(prep_attack, "comp_*.npy")):
        os.remove(f)
    for f in glob.glob(os.path.join(crops_dir, "*.jpg")):
        os.remove(f)

    made = 0
    while made < N_TARGET:
        p = patches[int(rng.integers(0, len(patches)))]
        patch = cv2.imread(p)
        if patch is None:
            continue
        if rng.random() < 0.5 and big_scenes:
            scene = cv2.imread(big_scenes[int(rng.integers(0, len(big_scenes)))])
            if scene is None:
                continue
            h, w = scene.shape[:2]
            side = int(min(h, w) * rng.uniform(0.10, 0.45))
        else:
            scene = cv2.imread(small_scenes[int(rng.integers(0, len(small_scenes)))])
            if scene is None:
                continue
            scene = cv2.resize(scene, (256, 256))
            h, w = scene.shape[:2]
            side = int(min(h, w) * rng.uniform(0.15, 0.45))
        if side < 12:
            continue
        pr = cv2.resize(patch, (side, side), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:  # lighting jitter like a live frame
            pr = np.clip(pr.astype(np.float32) * rng.uniform(0.75, 1.1), 0, 255).astype(np.uint8)
        y0, x0 = int(rng.integers(0, h - side)), int(rng.integers(0, w - side))
        comp = scene.copy()
        comp[y0:y0 + side, x0:x0 + side] = pr
        comp128 = cv2.resize(comp, (128, 128), interpolation=cv2.INTER_LINEAR)
        name = f"comp_{made:04d}"
        cv2.imwrite(os.path.join(crops_dir, name + ".jpg"), comp128)
        np.save(os.path.join(prep_attack, name + ".npy"), build_4ch_tensor(comp128))
        made += 1
    print(f"[comp] wrote {made} in-scene attack tensors -> {prep_attack}")


if __name__ == "__main__":
    main()
