"""
16_class_detection_rate.py
클래스별 탐지율 분석
소형 비율이 높은 클래스일수록 탐지율이 낮은지 확인
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ANNO     = "pest24_experiment/data/annotations/instances_test.json"
PRED_DIR = Path("pest24_experiment/results/predictions")
OUT_DIR  = Path("pest24_experiment/results/analysis")

with open(ANNO) as f:
    gt = json.load(f)

cat_info  = {c['id']: c['name'] for c in gt['categories']}
cat_ids   = sorted(cat_info.keys())
SMALL_AREA = 32 * 32

img_to_anns = defaultdict(list)
for ann in gt['annotations']:
    img_to_anns[ann['image_id']].append(ann)

MODELS = {
    "RT-DETR":      "rtdetr_predictions.json",
    "Faster R-CNN": "faster_rcnn_predictions.json",
    "YOLOv8":       "yolov8_predictions.json",
    "RetinaNet":    "retinanet_predictions.json",
}

IOU_THR = 0.5

def compute_class_detection(preds, gt_anns, cat_ids, small_area, iou_thr=0.5):
    pred_by_img = defaultdict(list)
    for p in preds:
        pred_by_img[p['image_id']].append(p)

    class_total       = defaultdict(int)
    class_tp          = defaultdict(int)
    class_small_total = defaultdict(int)
    class_small_tp    = defaultdict(int)

    for ann in gt_anns:
        cid    = ann['category_id']
        area   = ann['bbox'][2] * ann['bbox'][3]
        img_id = ann['image_id']
        gt_box = ann['bbox']

        class_total[cid] += 1
        is_small = area < small_area
        if is_small:
            class_small_total[cid] += 1

        img_preds = pred_by_img[img_id]
        matched = False
        for p in img_preds:
            if p['category_id'] != cid:
                continue
            pb = p['bbox']
            ix1 = max(gt_box[0], pb[0])
            iy1 = max(gt_box[1], pb[1])
            ix2 = min(gt_box[0]+gt_box[2], pb[0]+pb[2])
            iy2 = min(gt_box[1]+gt_box[3], pb[1]+pb[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2-ix1)*(iy2-iy1)
            union = gt_box[2]*gt_box[3] + pb[2]*pb[3] - inter
            iou   = inter/union if union > 0 else 0
            if iou >= iou_thr:
                matched = True
                break

        if matched:
            class_tp[cid] += 1
            if is_small:
                class_small_tp[cid] += 1

    result = {}
    for cid in cat_ids:
        total       = class_total[cid]
        tp          = class_tp[cid]
        small_total = class_small_total[cid]
        small_tp    = class_small_tp[cid]
        result[cid] = {
            'det_rate':       round(tp/total, 4) if total > 0 else 0,
            'small_det_rate': round(small_tp/small_total, 4) if small_total > 0 else None,
            'small_total':    small_total,
        }
    return result

# 모델별 클래스 탐지율 계산
all_results = {}
for model_name, fname in MODELS.items():
    fpath = PRED_DIR / fname
    if not fpath.exists():
        continue
    with open(fpath) as f:
        preds = json.load(f)
    all_results[model_name] = compute_class_detection(
        preds, gt['annotations'], cat_ids, SMALL_AREA
    )

# 소형 비율 불러오기
with open(OUT_DIR / "class_small_stats.json") as f:
    class_small_stats = json.load(f)
small_ratio = {s['id']: s['small_ratio'] for s in class_small_stats}

# 출력
print("\n클래스별 소형 객체 탐지율 (모델 비교)")
print(f"{'클래스':<35} {'소형비율':>8}", end="")
for name in MODELS:
    print(f" {name:>13}", end="")
print()
print("-" * (35 + 8 + 14*4))

# 소형 비율 높은 순으로 정렬
sorted_cats = sorted(cat_ids, key=lambda c: small_ratio.get(c, 0), reverse=True)

rows = []
for cid in sorted_cats:
    name  = cat_info[cid]
    sr    = small_ratio.get(cid, 0)
    row = {'id': cid, 'name': name, 'small_ratio': sr}
    print(f"{name:<35} {sr:>8.1%}", end="")
    for model_name in MODELS:
        r = all_results.get(model_name, {}).get(cid, {})
        sdr = r.get('small_det_rate')
        val = f"{sdr:.1%}" if sdr is not None else "  N/A"
        row[model_name] = sdr
        print(f" {val:>13}", end="")
    print()
    rows.append(row)

# 저장
with open(OUT_DIR / "class_detection_rates.json", 'w') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print("\n✅ 클래스별 탐지율 분석 완료!")
