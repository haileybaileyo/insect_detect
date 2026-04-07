"""
10_wbf_ensemble.py
4개 모델 WBF 앙상블
RT-DETR + YOLOv8 + Faster R-CNN + RetinaNet
"""
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

ANNO     = "pest24_experiment/data/annotations/instances_test.json"
PRED_DIR = Path("pest24_experiment/results/predictions")
OUT_PATH = PRED_DIR / "wbf_ensemble_v3_predictions.json"

MODELS = [
    ("rtdetr_predictions.json",       0.40),
    ("faster_rcnn_predictions.json",  0.30),
    ("yolov8_predictions.json",       0.15),
    ("retinanet_predictions.json",    0.15),
]

IOU_THR    = 0.5
SKIP_BOX   = 0.005
CONF_THR   = 0.20

print("=" * 60)
print(" WBF 앙상블 시작")
print(f" 모델: {[m[0] for m in MODELS]}")
print("=" * 60)

# GT 로드
with open(ANNO) as f:
    gt = json.load(f)

img_ids   = [img['id'] for img in gt['images']]
img_sizes = {img['id']: (img['width'], img['height']) for img in gt['images']}
num_cats  = len(gt['categories'])

# 예측 로드
all_preds = []
for fname, weight in MODELS:
    with open(PRED_DIR / fname) as f:
        all_preds.append(json.load(f))
    print(f"  로드: {fname} ({len(all_preds[-1])}개)")

# image_id별로 그룹화
from collections import defaultdict
model_by_img = []
for preds in all_preds:
    d = defaultdict(list)
    for p in preds:
        d[p['image_id']].append(p)
    model_by_img.append(d)

# WBF 앙상블
results = []
weights = [w for _, w in MODELS]

for img_id in tqdm(img_ids, desc="WBF 앙상블"):
    W, H = img_sizes[img_id]

    boxes_list, scores_list, labels_list = [], [], []

    for m_preds in model_by_img:
        preds = m_preds[img_id]
        boxes, scores, labels = [], [], []
        for p in preds:
            x, y, w, h = p['bbox']
            # WBF는 정규화된 [x1,y1,x2,y2] 필요
            x1 = max(0, x / W)
            y1 = max(0, y / H)
            x2 = min(1, (x + w) / W)
            y2 = min(1, (y + h) / H)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                scores.append(p['score'])
                labels.append(p['category_id'] - 1)  # 0-indexed

        boxes_list.append(boxes if boxes else [[0,0,0.001,0.001]])
        scores_list.append(scores if scores else [0.0])
        labels_list.append(labels if labels else [0])

    # WBF 실행
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights,
        iou_thr=IOU_THR,
        skip_box_thr=SKIP_BOX,
    )

    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        if score < CONF_THR:
            continue
        x1, y1, x2, y2 = box
        results.append({
            "image_id":    img_id,
            "category_id": int(label) + 1,  # 1-indexed
            "bbox":        [round(x1*W, 2), round(y1*H, 2),
                           round((x2-x1)*W, 2), round((y2-y1)*H, 2)],
            "score":       round(float(score), 4),
        })

with open(OUT_PATH, 'w') as f:
    json.dump(results, f)

print(f"\n  예측 수: {len(results)}")
print(f"  저장:    {OUT_PATH}")
print(f"\n✅ WBF 앙상블 완료!")
