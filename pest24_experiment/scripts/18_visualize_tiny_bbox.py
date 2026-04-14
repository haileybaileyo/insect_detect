"""
18_visualize_tiny_bbox.py
극소형 bbox 시각화 + Rice planthopper CV 분석 시각화
"""
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

ANNO     = "pest24_experiment/data/annotations/instances_train.json"
IMG_DIR  = Path("pest24_experiment/data/processed/train/images")
OUT_DIR  = Path("pest24_experiment/results/analysis/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ANNO) as f:
    gt = json.load(f)

cat_info = {c['id']: c['name'] for c in gt['categories']}
img_info = {img['id']: img for img in gt['images']}

# ─────────────────────────────
# 1. 극소형 bbox 샘플 시각화
# ─────────────────────────────
print("1. 극소형 bbox 시각화...")

tiny_anns = [ann for ann in gt['annotations']
             if ann['bbox'][2] * ann['bbox'][3] < 64]

print(f"  64px² 미만 bbox: {len(tiny_anns)}개")

saved = 0
for ann in tiny_anns[:12]:
    img_id = ann['image_id']
    fname  = img_info[img_id]['file_name']
    img_path = IMG_DIR / fname

    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    x, y, w, h = [int(v) for v in ann['bbox']]
    area = w * h
    cat_name = cat_info[ann['category_id']]

    # 전체 이미지에 bbox 표시
    img_full = img.copy()
    cv2.rectangle(img_full, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(img_full, f"{cat_name} {area}px²",
                (max(0,x), max(15,y-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    # crop (4배 확대)
    pad = 30
    x1 = max(0, x-pad); y1 = max(0, y-pad)
    x2 = min(img.shape[1], x+w+pad); y2 = min(img.shape[0], y+h+pad)
    crop = img[y1:y2, x1:x2]
    crop_big = cv2.resize(crop, (crop.shape[1]*4, crop.shape[0]*4),
                          interpolation=cv2.INTER_NEAREST)

    # 저장
    out_full = OUT_DIR / f"tiny_{saved:02d}_full_{cat_name.replace(' ','_')}.jpg"
    out_crop = OUT_DIR / f"tiny_{saved:02d}_crop_{cat_name.replace(' ','_')}.jpg"
    cv2.imwrite(str(out_full), img_full)
    cv2.imwrite(str(out_crop), crop_big)
    saved += 1

print(f"  저장: {saved}개 이미지")

# ─────────────────────────────
# 2. Rice planthopper 크기 분포 시각화
# ─────────────────────────────
print("\n2. Rice planthopper 크기 분포 시각화...")

rice_cat_id = [cid for cid, name in cat_info.items()
               if name == 'Rice planthopper'][0]
rice_areas = [ann['bbox'][2]*ann['bbox'][3]
              for ann in gt['annotations']
              if ann['category_id'] == rice_cat_id]

# 비교용: Gryllotalpa (가장 쉬운 클래스)
gryllo_cat_id = [cid for cid, name in cat_info.items()
                 if name == 'Gryllotalpa orientalis'][0]
gryllo_areas = [ann['bbox'][2]*ann['bbox'][3]
                for ann in gt['annotations']
                if ann['category_id'] == gryllo_cat_id]

# 히스토그램 그리기
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(rice_areas, bins=50, color='#E8593C', alpha=0.7, edgecolor='white')
axes[0].axvline(np.mean(rice_areas), color='red', linestyle='--',
                label=f'mean={np.mean(rice_areas):.0f}px²')
axes[0].axvline(1024, color='blue', linestyle=':', label='32² 임계점')
axes[0].set_title(f'Rice planthopper 크기 분포\n(n={len(rice_areas)}, CV={np.std(rice_areas)/np.mean(rice_areas):.2f})')
axes[0].set_xlabel('Bbox 면적 (px²)')
axes[0].set_ylabel('개수')
axes[0].legend()

axes[1].hist(gryllo_areas, bins=50, color='#1D9E75', alpha=0.7, edgecolor='white')
axes[1].axvline(np.mean(gryllo_areas), color='green', linestyle='--',
                label=f'mean={np.mean(gryllo_areas):.0f}px²')
axes[1].axvline(1024, color='blue', linestyle=':', label='32² 임계점')
axes[1].set_title(f'Gryllotalpa orientalis 크기 분포\n(n={len(gryllo_areas)}, CV={np.std(gryllo_areas)/np.mean(gryllo_areas):.2f})')
axes[1].set_xlabel('Bbox 면적 (px²)')
axes[1].set_ylabel('개수')
axes[1].legend()

plt.tight_layout()
plt.savefig(str(OUT_DIR / 'size_distribution_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  저장: size_distribution_comparison.png")

# ─────────────────────────────
# 3. 클래스별 평균 면적 vs 탐지율 관계
# ─────────────────────────────
print("\n3. 클래스별 평균 면적 vs 탐지율 산점도...")

# 탐지율 (RT-DETR 기준, 이전 분석 결과)
with open("pest24_experiment/results/analysis/class_detection_rates.json") as f:
    det_rates = json.load(f)

det_map = {d['name']: d.get('RT-DETR') for d in det_rates}

class_areas_mean = {}
for ann in gt['annotations']:
    cid = ann['category_id']
    name = cat_info[cid]
    if name not in class_areas_mean:
        class_areas_mean[name] = []
    class_areas_mean[name].append(ann['bbox'][2]*ann['bbox'][3])

fig, ax = plt.subplots(figsize=(10, 7))
for name, areas in class_areas_mean.items():
    det_rate = det_map.get(name)
    if det_rate is None:
        continue
    mean_area = np.mean(areas)
    ax.scatter(mean_area, det_rate, s=80, alpha=0.7,
               color='#E8593C' if mean_area < 1024 else '#1D9E75')
    ax.annotate(name, (mean_area, det_rate),
                fontsize=6, ha='left', va='bottom',
                xytext=(3, 3), textcoords='offset points')

ax.axvline(1024, color='blue', linestyle='--', alpha=0.5, label='32² 임계점')
ax.set_xlabel('클래스 평균 Bbox 면적 (px²)')
ax.set_ylabel('소형 객체 탐지율 (RT-DETR)')
ax.set_title('클래스 평균 면적 vs 소형 객체 탐지율\n(빨강: 소형 클래스, 초록: 중형+ 클래스)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT_DIR / 'area_vs_detection_rate.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  저장: area_vs_detection_rate.png")

print(f"\n✅ 시각화 완료! 저장: {OUT_DIR}")
