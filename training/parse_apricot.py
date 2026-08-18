"""
training/parse_apricot.py
Parses official APRICOT COCO annotations, crops physical adversarial patch ROIs (attack)
and background object ROIs (clean), and prepares 4-channel preprocessed tensors.
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
    clean_out = os.path.join(output_data_dir, "apde", "clean")
    attack_out = os.path.join(output_data_dir, "apde", "attack")
    prep_clean = os.path.join(output_data_dir, "preprocessed", "clean")
    prep_attack = os.path.join(output_data_dir, "preprocessed", "attack")

    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(attack_out, exist_ok=True)
    os.makedirs(prep_clean, exist_ok=True)
    os.makedirs(prep_attack, exist_ok=True)

    splits = [("dev", "apricot_annotations_dev.json"), ("test", "apricot_annotations_test.json")]
    
    total_attacks = 0
    total_cleans = 0

    print("=== Processing Real APRICOT Physical Adversarial Patch Dataset ===")

    for split_name, json_filename in splits:
        img_dir = os.path.join(apricot_dir, "Images", split_name)
        ann_path = os.path.join(apricot_dir, "Annotations", json_filename)

        if not os.path.exists(ann_path) or not os.path.exists(img_dir):
            print(f"[!] Skipping missing split: {split_name} ({ann_path})")
            continue

        print(f"[*] Parsing {split_name} annotations from {ann_path}...")
        with open(ann_path) as f:
            data = json.load(f)

        # Build image_id -> file_name mapping
        img_map = {img['id']: img['file_name'] for img in data.get('images', [])}
        
        # Group annotations by image_id
        anns_by_img = {}
        for ann in data.get('annotations', []):
            iid = ann['image_id']
            if iid not in anns_by_img:
                anns_by_img[iid] = []
            anns_by_img[iid].append(ann)

        print(f"[*] Extracting crops from {len(anns_by_img)} images in {split_name}...")
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
                bbox = ann.get('bbox') # [x, y, width, height]
                if not bbox or len(bbox) < 4:
                    continue

                bx, by, bw, bh = map(int, bbox)
                bx1 = max(0, bx)
                by1 = max(0, by)
                bx2 = min(w, bx + bw)
                by2 = min(h, by + bh)

                if bx2 <= bx1 or by2 <= by1 or (bx2 - bx1) < 10 or (by2 - by1) < 10:
                    continue

                roi = img[by1:by2, bx1:bx2]
                roi_resized = cv2.resize(roi, (128, 128))
                tensor_4ch = build_4ch_tensor(roi_resized)

                cat_id = ann.get('category_id')
                base_id = f"apricot_{split_name}_{img_id}_{idx}"

                if cat_id == 12: # Category 12 is 'adversarial patch'
                    cv2.imwrite(os.path.join(attack_out, f"{base_id}.jpg"), roi_resized)
                    np.save(os.path.join(prep_attack, f"{base_id}.npy"), tensor_4ch)
                    total_attacks += 1
                else: # Clean object crop
                    cv2.imwrite(os.path.join(clean_out, f"{base_id}.jpg"), roi_resized)
                    np.save(os.path.join(prep_clean, f"{base_id}.npy"), tensor_4ch)
                    total_cleans += 1

                # Extract a clean background crop outside the patch bbox for balance
                if cat_id == 12:
                    bg_y1 = max(0, by1 - bh if by1 - bh >= 0 else by2)
                    bg_y2 = min(h, bg_y1 + bh)
                    if bg_y2 > bg_y1 and bx2 > bx1:
                        bg_roi = img[bg_y1:bg_y2, bx1:bx2]
                        if bg_roi.shape[0] >= 10 and bg_roi.shape[1] >= 10:
                            bg_resized = cv2.resize(bg_roi, (128, 128))
                            bg_tensor = build_4ch_tensor(bg_resized)
                            cv2.imwrite(os.path.join(clean_out, f"{base_id}_bg.jpg"), bg_resized)
                            np.save(os.path.join(prep_clean, f"{base_id}_bg.npy"), bg_tensor)
                            total_cleans += 1

    print("\n=========================================================================")
    print("           REAL APRICOT DATASET EXTRACTION COMPLETE                     ")
    print("=========================================================================")
    print(f"  • Extracted Clean ROIs:       {total_cleans}")
    print(f"  • Extracted Physical Patches: {total_attacks}")
    print(f"  • Saved 4ch Tensors to:       data/preprocessed/")
    print("=========================================================================\n")


if __name__ == "__main__":
    parse_and_extract()
