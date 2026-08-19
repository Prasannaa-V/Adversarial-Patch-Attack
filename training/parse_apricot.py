"""
training/parse_apricot.py
Parses official APRICOT COCO annotations and crops physical adversarial patch ROIs (attack).
NO same-image clean background crops are generated to prevent data leakage.
Clean non-attack samples must be sourced from an independent dataset (COCO val2017).
"""

import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../inference'))
from sdnl import build_4ch_tensor


def parse_and_extract(apricot_dir="data/apricot", output_data_dir="data"):
    attack_out = os.path.join(output_data_dir, "crops", "attack")
    prep_attack = os.path.join(output_data_dir, "preprocessed", "attack")

    os.makedirs(attack_out, exist_ok=True)
    os.makedirs(prep_attack, exist_ok=True)

    splits = [("dev", "apricot_annotations_dev.json"), ("test", "apricot_annotations_test.json")]
    
    total_attacks = 0

    print("=== Extracting Real APRICOT Physical Adversarial Patch Crops ===")

    for split_name, json_filename in splits:
        img_dir = os.path.join(apricot_dir, "Images", split_name)
        ann_path = os.path.join(apricot_dir, "Annotations", json_filename)

        if not os.path.exists(ann_path) or not os.path.exists(img_dir):
            print(f"[!] Skipping missing split: {split_name} ({ann_path})")
            continue

        print(f"[*] Parsing {split_name} annotations from {ann_path}...")
        with open(ann_path) as f:
            data = json.load(f)

        img_map = {img['id']: img['file_name'] for img in data.get('images', [])}
        
        anns_by_img = {}
        for ann in data.get('annotations', []):
            if ann.get('category_id') == 12:  # Category 12 is 'adversarial patch'
                iid = ann['image_id']
                if iid not in anns_by_img:
                    anns_by_img[iid] = []
                anns_by_img[iid].append(ann)

        print(f"[*] Extracting physical patch crops from {len(anns_by_img)} images in {split_name}...")
        for img_id, anns in tqdm(anns_by_img.items()):
            file_name = img_map.get(img_id)
            if not file_name:
                continue
            
            img_path = os.path.join(img_dir, file_name)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            for idx, ann in enumerate(anns):
                bbox = ann.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue

                bx, by, bw_val, bh_val = map(int, bbox)
                bx1 = max(0, bx)
                by1 = max(0, by)
                bx2 = min(w, bx + bw_val)
                by2 = min(h, by + bh_val)

                if bx2 <= bx1 or by2 <= by1 or (bx2 - bx1) < 10 or (by2 - by1) < 10:
                    continue

                roi = img[by1:by2, bx1:bx2]
                roi_resized = cv2.resize(roi, (128, 128))
                tensor_4ch = build_4ch_tensor(roi_resized)

                base_id = f"apricot_{split_name}_{img_id}_{idx}"
                cv2.imwrite(os.path.join(attack_out, f"{base_id}.jpg"), roi_resized)
                np.save(os.path.join(prep_attack, f"{base_id}.npy"), tensor_4ch)
                total_attacks += 1

    print("\n=========================================================================")
    print("           REAL APRICOT PHYSICAL PATCH EXTRACTION COMPLETE               ")
    print("=========================================================================")
    print(f"  • Extracted Physical Patches: {total_attacks}")
    print(f"  • Saved 4ch Tensors to:       data/preprocessed/attack")
    print("  • NOTE: Zero same-image background crops were extracted to prevent leakage.")
    print("=========================================================================\n")


if __name__ == "__main__":
    parse_and_extract()
