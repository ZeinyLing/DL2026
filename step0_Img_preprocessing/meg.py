import cv2
import os
from tqdm import tqdm

# 原始影像資料夾
input_root = "./Tdata"
# 輸出資料夾
output_root = "./dataset_meg"

# CLAHE 設定
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 需要處理的三個 split
splits = ["train", "val", "test"]
classes = ["N", "P"]   # 子資料夾


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


for split in splits:
    for cls in classes:
        input_dir = os.path.join(input_root, split, cls)
        output_dir = os.path.join(output_root, split, cls)
        ensure_dir(output_dir)

        print(f"Processing: {input_dir}")

        for filename in tqdm(os.listdir(input_dir)):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                continue

            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print("Cannot read:", img_path)
                continue

            # 轉灰階 → CLAHE → 轉回 BGR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ---- CLAHE ----
            clahe_img = clahe.apply(gray)

            # ---- SOBEL ----
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = cv2.magnitude(sobel_x, sobel_y)
            sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
            sobel_mag = sobel_mag.astype("uint8")

            # ---- Stack 3 single-channel images into (H, W, 3) ----
            merged = cv2.merge([gray, clahe_img, sobel_mag])

            # ---- Save 3-channel image ----
            cv2.imwrite(os.path.join(output_dir, filename), merged)

print("Done! All images are processed.")
