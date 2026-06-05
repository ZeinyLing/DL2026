# -*- coding: utf-8 -*-
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.autograd import Variable


# ======================== 設定區 ===========================
DATA_DIR = "./dataset_ori/test/P/"       # 單一資料夾
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

OUT_CAM_DIR = "./cam_outputs/"
OUT_MASK_DIR = "./mask_outputs/"
os.makedirs(OUT_CAM_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tf_ops = [transforms.ToTensor()]
if USE_NORMALIZE:
    _tf_ops.append(
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    )
TO_TENSOR = transforms.Compose(_tf_ops)


# ======================== 基礎工具 ===========================
def load_model(path):
    mdl = torch.load(path, map_location=device)
    mdl.eval()
    return mdl.to(device)


def load_image_list(folder):
    return sorted([
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith(('.jpg','.png','.jpeg','.bmp','.tif'))
    ])


# ======================== 前處理版本產生 ===========================
def preprocess_all_versions(bgr):
    ori = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    clahe_f = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = clahe_f.apply(gray)
    clahe_3 = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    sobel_3 = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)

    meg = np.stack([gray, clahe, sobel], axis=2)

    # resize 全部
    ori = cv2.resize(ori, (IMG_SIZE, IMG_SIZE))
    gray_3 = cv2.resize(gray_3, (IMG_SIZE, IMG_SIZE))
    clahe_3 = cv2.resize(clahe_3, (IMG_SIZE, IMG_SIZE))
    meg = cv2.resize(meg, (IMG_SIZE, IMG_SIZE))
    sobel_3 = cv2.resize(sobel_3, (IMG_SIZE, IMG_SIZE))

    return ori, gray_3, clahe_3, meg, sobel_3


# ======================== CAM ===========================
def CAM_visualize(model, RGB_img, class_id):
    # 取得 FC 權重
    for name, parameters in model.named_parameters():
        if name == 'fc.weight':
            parm = parameters.cpu().detach().numpy()

    # 取得最後 conv 特徵
    model_act = nn.Sequential(*list(model.children())[:-2])
    model_act.eval()

    inp = transforms.ToTensor()(RGB_img).unsqueeze(0)
    if torch.cuda.is_available():
        inp = inp.cuda()

    with torch.no_grad():
        activation_maps = model_act(inp).cpu().numpy()[0]

    C, H, W = activation_maps.shape
    act_map = np.zeros((H, W), dtype=np.float32)

    # 加權合成 CAM
    for k in range(C):
        act_map += activation_maps[k] * parm[class_id][k]

    act_map -= act_map.min()
    act_map /= (act_map.max() + 1e-8)

    heatmap = cv2.resize(act_map, (IMG_SIZE, IMG_SIZE))

    # 色彩化
    hsv = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    hsv[:,:,0] = (1 - heatmap) * 120
    hsv[:,:,1] = 255
    hsv[:,:,2] = 255
    color_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    fusion = np.uint8(0.5 * RGB_img + 0.5 * color_map)

    return fusion, heatmap


def cam_to_mask(heatmap, thresh=0.4):
    mask = (heatmap >= thresh).astype(np.uint8) * 255
    return mask


# ======================== Voting ===========================
def predict_single(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    if float(conf) < THRESHOLD:
        return FALLBACK_CLASS
    return int(pred)


def weighted_hard_vote(pred_list, weights):
    num_classes = max(pred_list) + 1
    score = np.zeros(num_classes)
    for p, w in zip(pred_list, weights):
        score[p] += w
    return int(np.argmax(score))


# ======================== 主流程 ===========================
def main():
    t0 = time.time()

    models = [load_model(p) for p in MODEL_PATHS]
    imgs = load_image_list(DATA_DIR)

    csv_rows = []

    for img_path in imgs:
        name = os.path.basename(img_path)
        bgr = cv2.imread(img_path)

        # 產生五種輸入版本
        ori, g3, c3, meg, s3 = preprocess_all_versions(bgr)
        versions = [ori, g3, c3, meg, s3]

        # ----------------- 各版本模型預測 -----------------
        pred_list = []
        for v_img, model in zip(versions, models):
            tensor = TO_TENSOR(v_img).unsqueeze(0).to(device)
            pred = predict_single(model, tensor)
            pred_list.append(pred)

        voted = weighted_hard_vote(pred_list, WEIGHTS)

        # ----------------- CAM（僅模型預測與 VOTING 相同才做） -----------------
        cam_fusions = []
        cam_masks = []

        for k in range(len(models)):
            if pred_list[k] != voted:
                continue

            fusion, heatmap = CAM_visualize(models[k], versions[k], voted)
            mask = cam_to_mask(heatmap, thresh=0.4)

            cam_fusions.append(fusion)
            cam_masks.append(mask)

        # ----------------- 合併 CAM mask -----------------
        if len(cam_masks) > 0:
            # OR 合併
            final_mask = np.zeros_like(cam_masks[0])
            for m in cam_masks:
                final_mask = np.maximum(final_mask, m)

            # ===== 加入 DILATION (形態膨脹) =====
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        else:
            final_mask = None

        # ----------------- 輸出 CAM 與 mask -----------------
        if final_mask is not None:
            cv2.imwrite(f"{OUT_MASK_DIR}/{name[:-4]}.png", final_mask)
            cv2.imwrite(f"{OUT_CAM_DIR}/{name[:-4]}_cam.png",
                        cv2.cvtColor(cam_fusions[0], cv2.COLOR_RGB2BGR))

        # ----------------- 保存 CSV -----------------
        csv_rows.append([name, voted])

        print(f"[OK] {name} → voted={voted}")

    # ----------------- 輸出 CSV -----------------
    import csv
    with open("voting_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "voting_pred"])
        writer.writerows(csv_rows)

    print("\nCSV 已輸出：voting_results.csv")
    print(f"總耗時：{time.time() - t0:.2f} 秒")


if __name__ == "__main__":
    main()
