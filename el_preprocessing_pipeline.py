
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def preprocess_image(image):
    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image)

    # 2. Gaussian Blur
    image_blur = cv2.GaussianBlur(image_clahe, (3, 3), 0)

    # 3. Black-Hat Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(image_blur, cv2.MORPH_BLACKHAT, kernel)

    # 4. Sobel Edge Detection
    sobelx = cv2.Sobel(blackhat, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blackhat, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edge = cv2.magnitude(sobelx, sobely)
    sobel_edge = np.uint8(np.clip(sobel_edge, 0, 255))

    # 5. Normalize to 0–255
    normalized = cv2.normalize(sobel_edge, None, 0, 255, cv2.NORM_MINMAX)

    return normalized

def process_directory(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    for split in ['train', 'test']:
        split_input_path = input_root / split
        split_output_path = output_root / split

        for label_dir in split_input_path.iterdir():
            if label_dir.is_dir():
                output_label_dir = split_output_path / label_dir.name
                output_label_dir.mkdir(parents=True, exist_ok=True)

                for img_file in tqdm(list(label_dir.glob('*')), desc=f"Processing {label_dir}"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            processed = preprocess_image(image)
                            # print(f"Processed {img_file.name}")
                            save_path = output_label_dir / img_file.name
                            cv2.imwrite(str(save_path), processed)

if __name__ == "__main__":
    input_dir = "D:\\jungha\\2025_Spring\\MEC510\\term_project\\Processed_Data\\manmade"
    output_dir = "D:\\jungha\\2025_Spring\\MEC510\\term_project\\Processed_Data\\manmade_preprocessed"
    process_directory(input_dir, output_dir)
