"""
15_deep_analysis.py
다축 심층 분석
① 클래스별 AP_small
② 객체 크기 구간별 탐지율
③ 밀집도 구간별 성능
④ 공통 미탐지 패턴
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ANNO     = "pest24_experiment/data/annotations/instances_test.json"
PRED_DIR = Path("pest24_experiment/results/predictions")
OUT_DIR  = Path("pest24_experiment/results/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GT 로드
with open(ANNO) as f:
    gt = json.load(f)

# 기본 데이터 구조
img_info    = {img['id']: img for img in gt['images']}
cat_info    = {c['id']: c['name'] for c in gt['categories']}
cat_ids     = sorted(cat_info.keys())

# GT 어노테이션 정리
img_to_anns = defaultdict(list)
for ann in gt['annotations']:
    img_to_anns[ann['image_id']].append(ann)

# 소형 객체 기준
SMALL_AREA = 32 * 32

# ─────────────────────────────
# ① 클래스별 소형 객체 통계
# ─────────────────────────────
print("\n" + "="*60)
print("① 클래스별 소형 객체 통계")
print("="*60)

class_small_total = defaultdict(int)
class_total       = defaultdict(int)

for ann in gt['annotations']:
    cid = ann['category_id']
    area = ann['bbox'][2] * ann['bbox'][3]
    class_total[cid] += 1
    if area < SMALL_AREA:
        class_small_total[cid] += 1

class_stats = []
for cid in cat_ids:
    total = class_total[cid]
    small = class_small_total[cid]
    ratio = small / total if total > 0 else 0
    class_stats.append({
        'id': cid,
        'name': cat_info[cid],
        'total': total,
        'small': small,
        'small_ratio': round(ratio, 4)
    })

class_stats.sort(key=lambda x: x['small_ratio'], reverse=True)
print(f"{'클래스':<35} {'전체':>6} {'소형':>6} {'소형비율':>8}")
print("-" * 60)
for s in class_stats:
    print(f"{s['name']:<35} {s['total']:>6} {s['small']:>6} {s['small_ratio']:>8.1%}")

with open(OUT_DIR / "class_small_stats.json", 'w') as f:
    json.dump(class_stats, f, ensure_ascii=False, indent=2)

# ─────────────────────────────
# ② 객체 크기 구간별 GT 분포
# ─────────────────────────────
print("\n" + "="*60)
print("② 객체 크기 구간별 분포")
print("="*60)

size_bins = [
    (0,    64,   "극소형 (0~8²)"),
    (64,   256,  "소형   (8~16²)"),
    (256,  576,  "중소형 (16~24²)"),
    (576,  1024, "small상한 (24~32²)"),
    (1024, float('inf'), "중형+ (32²+)"),
]

size_counts = defaultdict(int)
for ann in gt['annotations']:
    area = ann['bbox'][2] * ann['bbox'][3]
    for lo, hi, label in size_bins:
        if lo <= area < hi:
            size_counts[label] += 1
            break

total_ann = len(gt['annotations'])
print(f"{'구간':<20} {'개수':>8} {'비율':>8}")
print("-" * 40)
for _, _, label in size_bins:
    cnt = size_counts[label]
    print(f"{label:<20} {cnt:>8} {cnt/total_ann:>8.1%}")

# ─────────────────────────────
# ③ 밀집도 구간별 이미지 분포
# ─────────────────────────────
print("\n" + "="*60)
print("③ 밀집도 구간별 이미지 분포")
print("="*60)

density_bins = [
    (1,  3,  "희소   (1~3개)"),
    (4,  6,  "보통   (4~6개)"),
    (7,  9,  "조밀   (7~9개)"),
    (10, float('inf'), "밀집   (10개+)"),
]

density_counts = defaultdict(int)
for img_id, anns in img_to_anns.items():
    n = len(anns)
    for lo, hi, label in density_bins:
        if lo <= n < hi or (hi == float('inf') and n >= lo):
            density_counts[label] += 1
            break

total_img = len(gt['images'])
print(f"{'구간':<20} {'이미지수':>8} {'비율':>8}")
print("-" * 40)
for _, _, label in density_bins:
    cnt = density_counts[label]
    print(f"{label:<20} {cnt:>8} {cnt/total_img:>8.1%}")

# ─────────────────────────────
# ④ 모델별 크기 구간별 탐지율
# ─────────────────────────────
print("\n" + "="*60)
print("④ 모델별 크기 구간별 탐지율")
print("="*60)

MODELS = {
    "RT-DETR":      "rtdetr_predictions.json",
    "Faster R-CNN": "faster_rcnn_predictions.json",
    "YOLOv8":       "yolov8_predictions.json",
    "RetinaNet":    "retinanet_predictions.json",
}

IOU_THR = 0.5

def compute_size_detection_rate(preds, gt_anns, size_bins, iou_thr=0.5):
    """크기 구간별 탐지율 계산"""
    from collections import defaultdict

    # 예측을 image_id별로 그룹화
    pred_by_img = defaultdict(list)
    for p in preds:
        pred_by_img[p['image_id']].append(p)

    bin_tp    = defaultdict(int)
    bin_total = defaultdict(int)

    for ann in gt_anns:
        area = ann['bbox'][2] * ann['bbox'][3]
        img_id = ann['image_id']
        gt_box = ann['bbox']  # [x,y,w,h]

        # 어느 구간인지
        bin_label = None
        for lo, hi, label in size_bins:
            if lo <= area < hi or (hi == float('inf') and area >= lo):
                bin_label = label
                break
        if bin_label is None:
            continue

        bin_total[bin_label] += 1

        # 이 GT와 매칭되는 예측 있는지 확인
        img_preds = pred_by_img[img_id]
        matched = False
        for p in img_preds:
            if p['category_id'] != ann['category_id']:
                continue
            pb = p['bbox']  # [x,y,w,h]
            # IoU 계산
            ix1 = max(gt_box[0], pb[0])
            iy1 = max(gt_box[1], pb[1])
            ix2 = min(gt_box[0]+gt_box[2], pb[0]+pb[2])
            iy2 = min(gt_box[1]+gt_box[3], pb[1]+pb[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2-ix1) * (iy2-iy1)
            union = gt_box[2]*gt_box[3] + pb[2]*pb[3] - inter
            iou   = inter / union if union > 0 else 0
            if iou >= iou_thr:
                matched = True
                break
        if matched:
            bin_tp[bin_label] += 1

    rates = {}
    for _, _, label in size_bins:
        total = bin_total[label]
        tp    = bin_tp[label]
        rates[label] = round(tp/total, 4) if total > 0 else 0
    return rates

print(f"\n{'구간':<22}", end="")
for name in MODELS:
    print(f"{name:>14}", end="")
print()
print("-" * (22 + 14*4))

all_rates = {}
for model_name, fname in MODELS.items():
    fpath = PRED_DIR / fname
    if not fpath.exists():
        print(f"  {model_name} 예측 파일 없음: {fname}")
        continue
    with open(fpath) as f:
        preds = json.load(f)
    rates = compute_size_detection_rate(preds, gt['annotations'], size_bins)
    all_rates[model_name] = rates

for _, _, label in size_bins:
    print(f"{label:<22}", end="")
    for model_name in MODELS:
        r = all_rates.get(model_name, {}).get(label, 0)
        print(f"{r:>14.1%}", end="")
    print()

# 저장
with open(OUT_DIR / "size_detection_rates.json", 'w') as f:
    json.dump(all_rates, f, ensure_ascii=False, indent=2)

# ─────────────────────────────
# ⑤ 밀집도 구간별 모델 성능
# ─────────────────────────────
print("\n" + "="*60)
print("⑤ 밀집도 구간별 모델 탐지율")
print("="*60)

def compute_density_detection_rate(preds, img_to_anns, density_bins, iou_thr=0.5):
    pred_by_img = defaultdict(list)
    for p in preds:
        pred_by_img[p['image_id']].append(p)

    bin_tp    = defaultdict(int)
    bin_total = defaultdict(int)

    for img_id, anns in img_to_anns.items():
        n = len(anns)
        bin_label = None
        for lo, hi, label in density_bins:
            if lo <= n < hi or (hi == float('inf') and n >= lo):
                bin_label = label
                break
        if bin_label is None:
            continue

        img_preds = pred_by_img[img_id]
        for ann in anns:
            bin_total[bin_label] += 1
            matched = False
            for p in img_preds:
                if p['category_id'] != ann['category_id']:
                    continue
                pb = p['bbox']
                gt_box = ann['bbox']
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
                bin_tp[bin_label] += 1

    rates = {}
    for _, _, label in density_bins:
        total = bin_total[label]
        tp    = bin_tp[label]
        rates[label] = round(tp/total, 4) if total > 0 else 0
    return rates

print(f"\n{'구간':<22}", end="")
for name in MODELS:
    print(f"{name:>14}", end="")
print()
print("-" * (22 + 14*4))

density_rates = {}
for model_name, fname in MODELS.items():
    fpath = PRED_DIR / fname
    if not fpath.exists():
        continue
    with open(fpath) as f:
        preds = json.load(f)
    rates = compute_density_detection_rate(preds, img_to_anns, density_bins)
    density_rates[model_name] = rates

for _, _, label in density_bins:
    print(f"{label:<22}", end="")
    for model_name in MODELS:
        r = density_rates.get(model_name, {}).get(label, 0)
        print(f"{r:>14.1%}", end="")
    print()

with open(OUT_DIR / "density_detection_rates.json", 'w') as f:
    json.dump(density_rates, f, ensure_ascii=False, indent=2)

print("\n✅ 심층 분석 완료!")
print(f"결과 저장: {OUT_DIR}")
