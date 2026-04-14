"""
17_annotation_quality.py
어노테이션 품질 분석
1. bbox 크기 분포
2. bbox 종횡비 분포
3. 이미지 경계 초과 bbox
4. 극소형 bbox (노이즈 가능성)
5. 클래스별 bbox 크기 일관성
6. 샘플 이미지 시각화
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ANNO    = "pest24_experiment/data/annotations/instances_train.json"
OUT_DIR = Path("pest24_experiment/results/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ANNO) as f:
    gt = json.load(f)

cat_info    = {c['id']: c['name'] for c in gt['categories']}
img_info    = {img['id']: img for img in gt['images']}
annotations = gt['annotations']

print(f"총 어노테이션: {len(annotations)}개")
print(f"총 이미지:    {len(gt['images'])}개")

# ─────────────────────────────────────────────
# 1. bbox 기본 통계
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("1. Bbox 크기 기본 통계")
print("="*60)

areas, widths, heights, ratios = [], [], [], []
zero_area, negative, out_of_bound = 0, 0, 0

for ann in annotations:
    x, y, w, h = ann['bbox']
    img = img_info[ann['image_id']]
    IW, IH = img['width'], img['height']

    # 비정상 bbox 체크
    if w <= 0 or h <= 0:
        negative += 1
        continue
    if x < 0 or y < 0 or x+w > IW or y+h > IH:
        out_of_bound += 1

    area = w * h
    if area == 0:
        zero_area += 1
        continue

    areas.append(area)
    widths.append(w)
    heights.append(h)
    ratios.append(w / h)

areas   = np.array(areas)
widths  = np.array(widths)
heights = np.array(heights)
ratios  = np.array(ratios)

print(f"비정상 bbox (w/h ≤ 0): {negative}개")
print(f"경계 초과 bbox:         {out_of_bound}개  ({out_of_bound/len(annotations)*100:.1f}%)")
print(f"면적 0 bbox:            {zero_area}개")
print()
print(f"bbox 면적 통계:")
print(f"  min:    {areas.min():.1f} px²")
print(f"  max:    {areas.max():.1f} px²")
print(f"  mean:   {areas.mean():.1f} px²")
print(f"  median: {np.median(areas):.1f} px²")
print(f"  std:    {areas.std():.1f} px²")
print()
print(f"bbox 너비 통계:")
print(f"  min: {widths.min():.1f}  max: {widths.max():.1f}  mean: {widths.mean():.1f}")
print(f"bbox 높이 통계:")
print(f"  min: {heights.min():.1f}  max: {heights.max():.1f}  mean: {heights.mean():.1f}")

# ─────────────────────────────────────────────
# 2. 극소형 bbox 분포 (노이즈 가능성)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("2. 극소형 Bbox 분포 (노이즈 가능성)")
print("="*60)

thresholds = [4, 9, 16, 25, 36, 64, 100, 256]
print(f"{'면적 < X px²':<15} {'개수':>8} {'비율':>8}")
print("-"*35)
for thr in thresholds:
    cnt = (areas < thr).sum()
    print(f"< {thr:>5} px²      {cnt:>8} {cnt/len(areas)*100:>8.2f}%")

# 특히 1픽셀 이하
tiny = (areas < 4).sum()
print(f"\n2×2px 미만 (노이즈 의심): {tiny}개")

# ─────────────────────────────────────────────
# 3. 종횡비 분포 (극단적 비율 = 오류 가능성)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("3. 종횡비(w/h) 분포")
print("="*60)

ratio_bins = [
    (0, 0.1,  "극단적 세로 (< 0.1)"),
    (0.1, 0.3, "세로 (0.1~0.3)"),
    (0.3, 3.0, "정상 (0.3~3.0)"),
    (3.0, 10, "가로 (3~10)"),
    (10, 999, "극단적 가로 (> 10)"),
]
for lo, hi, label in ratio_bins:
    cnt = ((ratios >= lo) & (ratios < hi)).sum()
    print(f"{label:<25} {cnt:>8} {cnt/len(ratios)*100:>8.1f}%")

# ─────────────────────────────────────────────
# 4. 클래스별 bbox 크기 일관성
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("4. 클래스별 Bbox 크기 통계 (평균 면적)")
print("="*60)

class_areas = defaultdict(list)
for ann in annotations:
    x, y, w, h = ann['bbox']
    if w > 0 and h > 0:
        class_areas[ann['category_id']].append(w * h)

class_stats = []
for cid, area_list in class_areas.items():
    arr = np.array(area_list)
    class_stats.append({
        'name': cat_info[cid],
        'count': len(arr),
        'mean_area': arr.mean(),
        'std_area': arr.std(),
        'cv': arr.std() / arr.mean() if arr.mean() > 0 else 0,
        'min': arr.min(),
        'max': arr.max(),
    })

class_stats.sort(key=lambda x: x['mean_area'])
print(f"{'클래스':<35} {'샘플':>6} {'평균면적':>10} {'std':>10} {'CV':>6} {'min':>8} {'max':>8}")
print("-"*90)
for s in class_stats:
    print(f"{s['name']:<35} {s['count']:>6} {s['mean_area']:>10.1f} "
          f"{s['std_area']:>10.1f} {s['cv']:>6.2f} {s['min']:>8.1f} {s['max']:>8.1f}")

# CV(변동계수)가 높은 클래스 = 크기 일관성이 낮음 = 어노테이션 오류 가능성
print("\nCV(변동계수) 상위 5개 클래스 (크기 일관성 낮음 = 오류 가능성):")
for s in sorted(class_stats, key=lambda x: x['cv'], reverse=True)[:5]:
    print(f"  {s['name']}: CV={s['cv']:.2f} (면적 {s['min']:.0f}~{s['max']:.0f})")

# ─────────────────────────────────────────────
# 5. 중복 어노테이션 검출
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("5. 중복 어노테이션 검출 (같은 위치 bbox)")
print("="*60)

from collections import defaultdict
img_ann_map = defaultdict(list)
for ann in annotations:
    img_ann_map[ann['image_id']].append(ann)

duplicate_count = 0
for img_id, anns in img_ann_map.items():
    for i in range(len(anns)):
        for j in range(i+1, len(anns)):
            b1 = anns[i]['bbox']
            b2 = anns[j]['bbox']
            if anns[i]['category_id'] != anns[j]['category_id']:
                continue
            # IoU 계산
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[0]+b1[2], b2[0]+b2[2])
            iy2 = min(b1[1]+b1[3], b2[1]+b2[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2-ix1)*(iy2-iy1)
            union = b1[2]*b1[3] + b2[2]*b2[3] - inter
            iou = inter/union if union > 0 else 0
            if iou > 0.9:
                duplicate_count += 1

print(f"IoU > 0.9 중복 어노테이션: {duplicate_count}개")

# ─────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────
import json
result = {
    'total_annotations': len(annotations),
    'negative_bbox': int(negative),
    'out_of_bound': int(out_of_bound),
    'zero_area': int(zero_area),
    'area_stats': {
        'min': float(areas.min()),
        'max': float(areas.max()),
        'mean': float(areas.mean()),
        'median': float(np.median(areas)),
        'std': float(areas.std()),
    },
    'tiny_bbox_count': {str(thr): int((areas < thr).sum()) for thr in thresholds},
    'duplicate_count': duplicate_count,
    'class_stats': class_stats,
}
with open(OUT_DIR / 'annotation_quality.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n✅ 분석 완료! 결과 저장: {OUT_DIR}/annotation_quality.json")
