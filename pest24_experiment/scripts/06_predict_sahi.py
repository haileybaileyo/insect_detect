"""
06_predict_sahi.py
RT-DETR + SAHI 파이프라인 추론
→ 슬라이싱으로 소형 객체 탐지 개선

실행:
  python3 pest24_experiment/scripts/06_predict_sahi.py
"""

import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
CKPT     = "pest24_experiment/checkpoints/rtdetr/rtdetr_l_imgsz6409/weights/best.pt"
ANNO     = "pest24_experiment/data/annotations/instances_test.json"
IMG_DIR  = Path("pest24_experiment/data/processed/test/images")
PRED_DIR = Path("pest24_experiment/results/predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

CONF            = 0.25
SLICE_SIZE      = 640
OVERLAP         = 0.2
MATCH_THRESHOLD = 0.3   # NMM threshold (기본 0.5 → 0.3으로 낮춤)
MODEL_TYPE  = "ultralytics"

# ─────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────
print("=" * 60)
print(" RT-DETR + SAHI 추론 시작")
print(f" 슬라이스 크기: {SLICE_SIZE}×{SLICE_SIZE}, overlap={OVERLAP}")
print("=" * 60)

detection_model = AutoDetectionModel.from_pretrained(
    model_type=MODEL_TYPE,
    model_path=CKPT,
    confidence_threshold=CONF,
    device="cuda:0",
)

# ─────────────────────────────────────────────
# GT 정보 로드
# ─────────────────────────────────────────────
with open(ANNO) as f:
    gt = json.load(f)

fname_to_id = {Path(img['file_name']).name: img['id'] for img in gt['images']}
cat_ids     = sorted([c['id'] for c in gt['categories']])
yolo_to_cat = {idx: cid for idx, cid in enumerate(cat_ids)}

img_list = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
print(f"  테스트 이미지: {len(img_list)}")

# ─────────────────────────────────────────────
# SAHI 추론
# ─────────────────────────────────────────────
predictions = []
fps_times   = []

for img_path in tqdm(img_list, desc="  SAHI 추론"):
    fname = img_path.name
    if fname not in fname_to_id:
        continue
    img_id = fname_to_id[fname]

    t0 = time.perf_counter()
    result = get_sliced_prediction(
        str(img_path),
        detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        postprocess_type="NMM",
        postprocess_match_threshold=MATCH_THRESHOLD,
        verbose=0,
    )
    t1 = time.perf_counter()
    fps_times.append(t1 - t0)

    for obj in result.object_prediction_list:
        bbox = obj.bbox
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        cat_id = yolo_to_cat.get(obj.category.id, obj.category.id + 1)
        predictions.append({
            "image_id":    img_id,
            "category_id": cat_id,
            "bbox":        [round(x1,2), round(y1,2),
                            round(x2-x1,2), round(y2-y1,2)],
            "score":       round(obj.score.value, 4),
        })

# ─────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────
out_path = PRED_DIR / "rtdetr_sahi_nmm03_predictions.json"
with open(out_path, 'w') as f:
    json.dump(predictions, f)

fps_times = fps_times[5:]
avg_ms = np.mean(fps_times) * 1000
fps    = 1.0 / np.mean(fps_times)

fps_meta = {
    "model":           "rtdetr_sahi",
    "num_predictions": len(predictions),
    "fps":             round(fps, 2),
    "ms_per_img":      round(avg_ms, 2),
    "slice_size":      SLICE_SIZE,
    "overlap":         OVERLAP,
}
with open(PRED_DIR / "rtdetr_sahi_fps.json", 'w') as f:
    json.dump(fps_meta, f, indent=2)

print(f"\n  FPS:      {fps:.1f} ({avg_ms:.1f} ms/img)")
print(f"  예측 수:  {len(predictions)}")
print(f"  저장:     {out_path}")
print(f"\n✅ RT-DETR + SAHI 추론 완료!")
