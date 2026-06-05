# -*- coding: utf-8 -*-
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score

# ======================== 設定區 ===========================
DATA_DIR = "./results/"  # 單一資料夾，裡面只有影像
IMG_SIZE = 256

MODEL_PATHS = [
    "./resnet50_ep.pkl",             # ORI 模型
    "./resnet50_ep_100_gray.pkl",    # GRAY 模型
    "./resnet50_ep_100_clahe.pkl",   # CLAHE 模型
    "./resnet50_ep_100_meg.pkl",     # MEG 模型
    "./resnet50_ep_100_sobel.pkl",   # SOBEL 模型
]

USE_NORMALIZE = False
THRESHOLD = 0.99
FALLBACK_CLASS = 0
WEIGHTS = [0.65, 0.1, 0.1, 0.1, 0.05]
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tf_ops = [transforms.ToTensor()]
if USE_NORMALIZE:
    _tf_ops.append(transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]))
TO_TENSOR = transforms.Compose(_tf_ops)


# -----------------------------------------------------------
def load_model(path):
    mdl = torch.load(path, map_location=device)
    mdl.eval()
    return mdl.to(device)


# -----------------------------------------------------------
def load_image_list(folder):
    paths = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(('.jpg','.png','.jpeg','.bmp','.tif')):
            paths.append(os.path.join(folder, fn))
    return paths


# -----------------------------------------------------------
def preprocess_all_versions(bgr):
    """
    回傳：
    ori_rgb, gray_3ch, clahe_3ch, meg_3ch, sobel_3ch
    """

    # ORI
    ori_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # GRAY
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # CLAHE
    clahe_f = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = clahe_f.apply(gray)
    clahe_3ch = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)

    # SOBEL
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    sobel_3ch = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)

    # MEG = Gray, CLAHE, SOBEL 疊成 3 channel
    meg_3ch = np.stack([gray, clahe, sobel], axis=2)

    # resize 全部
    ori_rgb = cv2.resize(ori_rgb, (IMG_SIZE, IMG_SIZE))
    gray_3ch = cv2.resize(gray_3ch, (IMG_SIZE, IMG_SIZE))
    clahe_3ch = cv2.resize(clahe_3ch, (IMG_SIZE, IMG_SIZE))
    meg_3ch = cv2.resize(meg_3ch, (IMG_SIZE, IMG_SIZE))
    sobel_3ch = cv2.resize(sobel_3ch, (IMG_SIZE, IMG_SIZE))

    return ori_rgb, gray_3ch, clahe_3ch, meg_3ch, sobel_3ch


# -----------------------------------------------------------
def predict_single(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    conf = float(conf.item())
    pred = int(pred.item())

    if conf < THRESHOLD:
        return FALLBACK_CLASS
    return pred


# -----------------------------------------------------------
def weighted_hard_vote(pred_list, weights):
    num_classes = max(pred_list) + 1
    score = np.zeros(num_classes, dtype=float)
    for p, w in zip(pred_list, weights):
        score[p] += w
    return int(np.argmax(score))


# -----------------------------------------------------------
def main():
    t0 = time.time()

    models = [load_model(p) for p in MODEL_PATHS]
    img_list = load_image_list(DATA_DIR)

    print(f"總圖片：{len(img_list)}")

    voting_preds = []
    img_names = []

    for img_path in img_list:
        bgr = cv2.imread(img_path)

        ori, g3, c3, meg, s3 = preprocess_all_versions(bgr)
        versions = [ori, g3, c3, meg, s3]

        pred_list = []

        for v_img, model in zip(versions, models):
            tensor = TO_TENSOR(v_img).unsqueeze(0).to(device)
            pred = predict_single(model, tensor)
            pred_list.append(pred)

        voted = weighted_hard_vote(pred_list, WEIGHTS)
        voting_preds.append(voted)

        img_names.append(os.path.basename(img_path))
    print(voting_preds.count(1))
    # -----------------------------
    # 匯出 CSV
    # -----------------------------
    import csv
    csv_path = "voting_resultsNEW.csv"
    header = ["image_name", "voting_pred"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, pred in zip(img_names, voting_preds):
            writer.writerow([name, pred])

    print(f"\nCSV 已輸出：{csv_path}")
    print(f"總耗時：{time.time() - t0:.2f} 秒")


if __name__ == "__main__":
    main()
