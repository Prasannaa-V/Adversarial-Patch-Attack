"""Fetch external CLEAN negatives from the HuggingFace INRIA-person mirror
(marcelarosalesj/inria-person): real 2005-era pedestrian photos, a camera
distribution our COCO/synth clean set has never seen.

Saves JPGs to data/inria_clean/ + 4-channel tensors to
data/preprocessed/ext_inria/*.npy (kept OUTSIDE data/preprocessed/clean/
so the training pool is untouched; external eval only).
Requires: pip install datasets huggingface_hub pillow
"""
import os
import sys

N_TARGET = 300

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from sdnl import build_4ch_tensor


def main():
    import numpy as np
    from datasets import load_dataset

    img_dir = os.path.join("data", "inria_clean", "images")
    prep_dir = os.path.join("data", "preprocessed", "ext_inria")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)

    ds = load_dataset("marcelarosalesj/inria-person", split="train")
    print(f"[inria] mirror rows: {len(ds)}; features: {ds.features}")
    made = 0
    for i, row in enumerate(ds):
        if made >= N_TARGET:
            break
        try:
            img = row["image"]
            arr = np.array(img.convert("RGB"))
        except Exception as e:
            print(f"[inria] skip row {i}: {e}")
            continue
        h, w = arr.shape[:2]
        if h < 128 or w < 128:
            continue
        import cv2
        # center crop (pedestrian mirrors are usually centered) + corner crop
        for tag, crop in (("c", arr[h // 2 - 64:h // 2 + 64, w // 2 - 64:w // 2 + 64]),
                          ("t", arr[0:128, 0:128])):
            if crop.shape[:2] != (128, 128):
                continue
            bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            name = f"ext_inria_{made:04d}{tag}"
            cv2.imwrite(os.path.join(img_dir, name + ".jpg"), bgr)
            np.save(os.path.join(prep_dir, name + ".npy"), build_4ch_tensor(bgr))
            made += 1
    print(f"[inria] wrote {made} external clean tensors -> {prep_dir}")


if __name__ == "__main__":
    main()
