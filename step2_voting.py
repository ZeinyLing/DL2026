# -*- coding: utf-8 -*-
import os
import re
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix
import torch.nn.functional as F

# ======================== 設定區 ===========================
DATA_DIRS = [
    "./dataset_ori/test/",
    "./dataset_gray/test/",
    "./dataset_CLAHE/test/",
    "./dataset_meg/test/",
    "./dataset_sobel/test/",
]

IMG_SIZE = 256

MODEL_PATHS = [
    "./resnet50_ep.pkl",
    "./resnet50_ep_100_gray.pkl",
    "./resnet50_ep_100_clahe.pkl",
    "./resnet50_ep_100_meg.pkl",
    "./resnet50_ep_100_sobel.pkl",
]

USE_NORMALIZE = False
SAVE_FIGS = False
THRESHOLD = 0.99
FALLBACK_CLASS = 0    # 低於門檻時預設類別
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
def list_images_and_labels(root):
    classes = sorted([d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d))])
    samples = []
    for ci, cname in enumerate(classes):
        cdir = os.path.join(root, cname)
        for fn in os.listdir(cdir):
            if fn.lower().endswith(('.jpg','.png','.jpeg','.bmp','.tif')):
                samples.append((os.path.join(cdir, fn), ci))
    return samples, classes


# -----------------------------------------------------------
def resize_to_rgb(img_bgr, size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(img_rgb, (size, size))


# -----------------------------------------------------------
def predict_single(model, tensor, threshold):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    conf = float(conf.item())
    pred = int(pred.item())

    if conf < threshold:
        return FALLBACK_CLASS

    return pred


# -----------------------------------------------------------
def hard_vote(pred_list):
    """ pred_list: [pred_A, pred_B, pred_C, ...] """
    return int(np.bincount(pred_list).argmax())


# -----------------------------------------------------------
def evaluate(true_labels, pred_labels, num_classes):
    avg = "binary" if num_classes == 2 else "macro"

    prec = precision_score(true_labels, pred_labels, average=avg)
    rec  = recall_score(true_labels, pred_labels, average=avg)
    f1   = f1_score(true_labels, pred_labels, average=avg)

    return prec, rec, f1

def weighted_soft_vote(probs_list, weights):
    """
    probs_list: [prob_A (C-dim), prob_B, prob_C]
    weights:    [wA, wB, wC]
    """
    weights = np.array(weights)
    weights = weights / weights.sum()  # normalize

    combined = np.zeros_like(probs_list[0])

    for prob, w in zip(probs_list, weights):
        combined += prob * w

    return int(np.argmax(combined))
    
def weighted_hard_vote(pred_list, weights):
    """
    pred_list: [pred_A, pred_B, pred_C]
    weights:   [wA, wB, wC]
    """
    num_classes = max(pred_list) + 1
    score = np.zeros(num_classes, dtype=float)

    for p, w in zip(pred_list, weights):
        score[p] += w

    return int(np.argmax(score))

# -----------------------------------------------------------
def main():
    t0 = time.time()

    assert len(DATA_DIRS) == len(MODEL_PATHS), \
        "資料集與模型必須一對一：DATA_DIRS[i] 對應 MODEL_PATHS[i]"

    K = len(DATA_DIRS)
    models = [load_model(p) for p in MODEL_PATHS]

    # ==== 確保所有資料集圖片排序一致 ====
    samples0, class_names = list_images_and_labels(DATA_DIRS[0])
    all_paths = [s[0] for s in samples0]
    all_true = [s[1] for s in samples0]
    num_classes = len(class_names)

    print(f"資料集總圖片：{len(all_paths)}")

    # 每個模型對應自己的 dataset
    all_model_preds = []

    for di in range(K):
        data_dir = DATA_DIRS[di]
        model = models[di]

        print(f"\n=== Dataset {di+1}: {data_dir}")
        print(f"=== Model   {di+1}: {MODEL_PATHS[di]}")

        samples, _ = list_images_and_labels(data_dir)
        samples = sorted(samples, key=lambda x: x[0])  # 強制排序一致

        preds = []

        for (img_path, true_idx), std_path in zip(samples, all_paths):
            bgr = cv2.imread(img_path)
            rgb = resize_to_rgb(bgr, IMG_SIZE)
            tensor = TO_TENSOR(rgb).unsqueeze(0).to(device)

            pred = predict_single(model, tensor, THRESHOLD)
            preds.append(pred)

        all_model_preds.append(preds)

    # -----------------------------
    # 最終 Voting
    # -----------------------------
    WEIGHTS = [0.65, 0.1, 0.1, 0.1, 0.05]

    voting_preds = []
    for i in range(len(all_paths)):
        pred_list = [all_model_preds[k][i] for k in range(K)]
        voted = weighted_hard_vote(pred_list, WEIGHTS)
        voting_preds.append(voted)

    # === 評估 Voting 結果 ===
    prec, rec, f1 = evaluate(all_true, voting_preds, num_classes)

    print("\n========== Voting 結果 ==========")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print("================================\n")

    print(f"總耗時：{time.time() - t0:.2f} 秒")
    cm = confusion_matrix(all_true, voting_preds, labels=list(range(num_classes)))

    print("Confusion Matrix:")
    print(cm)
        # -----------------------------
    # 匯出 CSV：只保存 Voting 結果
    # -----------------------------
    import csv

    csv_path = "voting_only_results.csv"
    header = ["image_path", "true_label", "voting_pred"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, img_path in enumerate(all_paths):
            img_path = img_path.split('/')[-1]
            row = [
                img_path,         # 圖片路徑
                all_true[idx],    # 真實標籤
                voting_preds[idx] # Voting 最後預測
            ]
            writer.writerow(row)

    print(f"\nCSV 已輸出：{csv_path}\n")

if __name__ == "__main__":
    main()