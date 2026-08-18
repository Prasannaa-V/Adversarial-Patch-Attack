import cv2
import numpy as np
import os

def generate_dummy_images(output_dir, num_images=10):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_images):
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"img_{i}.jpg"), img)

if __name__ == "__main__":
    generate_dummy_images("data/apde/clean", 50)
    generate_dummy_images("data/apde/attack", 50)
    print("Generated dummy images.")
