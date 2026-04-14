"""
18_predict_rtdetr_896.py
RT-DETR 896 test 셋 추론
"""
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import RTDETR

CKPT     = "runs/detect/pest24_experiment/checkpoints/rtdetr_896/rtdetr_l_imgsz896/weights/best.pt"
ANNO     = "pest24_experiment/data/annotations/instances_test.json"
IMG_DIR  = Path("pest24_experiment/data/processed/test/images")
PRED_DIR = Path("pest24_experiment/results/predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

CONF   = 0.25
IMGSZ  = 896

with open(ANNO) as f:
    gt = json.load(f)
fname_to_id = {Path(img['file_name']).name: img['id'] for img in gt['images']}
cat_ids     = sorted([c['id'] for c in gt['categories']])
yolo_to_cat = {idx: cid for idx, cid in enumerate(cat_ids)}
img_list    = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))

print(f"RT-DETR 896 추론 시작 ({len(img_list)}장)")
model = RTDETR(CKPT)
predictions = []
fps_times   = []

for img_path in tqdm(img_list, desc="  추론"):
    fname = img_path.name
    if fname not in fname_to_id:
        continue
    img_id = fname_to_id[fname]
    t0 = time.perf_counter()
    results = model.predict(str(img_path), imgsz=IMGSZ, conf=CONF, device=0, verbose=False)
    t1 = time.perf_counter()
    fps_times.append(t1 - t0)
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx = int(box.cls[0])
            score   = float(box.conf[0])
            cat_id  = yolo_to_cat.get(cls_idx, cls_idx + 1)
            predictions.append({
                "image_id":    img_id,
                "category_id": cat_id,
                "bbox":        [round(x1,2), round(y1,2), round(x2-x1,2), round(y2-y1,2)],
                "score":       round(score, 4),
            })

out_path = PRED_DIR / "rtdetr_896_predictions.json"
with open(out_path, 'w') as f:
    json.dump(predictions, f)

fps = 1.0 / np.mean(fps_times[5:])
print(f"\n  예측 수: {len(predictions)}")
print(f"  FPS:    {fps:.1f}")
print(f"  저장:   {out_path}")
print("✅ 완료!")
