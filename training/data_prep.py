"""
training/data_prep.py
Preprocesses training dataset (APDE / custom) to extract 4-channel tensors:
  Channels 1-3: RGB ROI (128x128)
  Channel 4:   DCT high-frequency map (128x128)

Saves as preprocessed numpy files for fast training.
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import glob

# Add inference directory to path to reuse build_4ch_tensor
sys.path.append(os.path.join(os.path.dirname(__file__), '../inference'))
from sdnl import build_4ch_tensor


def preprocess_dataset(input_dir: str, output_dir: str, label_name: str):
    """
    Reads all images in input_dir, builds 4-channel tensors, 
    and saves them to output_dir/<label_name> as .npy files.
    """
    os.makedirs(os.path.join(output_dir, label_name), exist_ok=True)
    
    # Supported formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.PNG', '*.JPG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

    print(f"Found {len(image_paths)} images in '{input_dir}' for label '{label_name}'")
    
    count = 0
    for path in tqdm(image_paths):
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            
            # If the raw image is not already a crop of the person/vehicle ROI,
            # this script assumes input images are already pre-cropped ROI boxes 
            # (which is typical for patch training datasets).
            tensor = build_4ch_tensor(img)
            
            # Save preprocessed tensor
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, label_name, f"{base}.npy")
            np.save(out_path, tensor)
            count += 1
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    print(f"Successfully preprocessed {count} images for label '{label_name}'")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset to 4-channel tensors")
    parser.add_argument("--clean-dir", type=str, required=True, help="Directory of clean person/vehicle crops")
    parser.add_argument("--attack-dir", type=str, required=True, help="Directory of adversarial patch crops")
    parser.add_argument("--output-dir", type=str, default="data/preprocessed", help="Output directory")
    args = parser.parse_args()

    print("=== InfraEdge-Shield Data Preprocessing ===")
    preprocess_dataset(args.clean_dir, args.output_dir, "clean")
    preprocess_dataset(args.attack_dir, args.output_dir, "attack")
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
