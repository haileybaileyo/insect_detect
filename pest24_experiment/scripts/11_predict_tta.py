"""
11_predict_tta.py
RT-DETR TTA 추론
480, 800 스케일 추론 후 저장
"""
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import RTDETR

CKPT     = "pest24_experiment/checkpoints/rtdetr/rtdetr_l_imgsz6409/weights/best.pt"
ANNO     = "pest24_experiment/data/annotations/instances_test.json"
IMG_DIR  = Path("pest24_experiment/data/processed/test/images")
PRED_DIR = Path("pest24_experiment/results/predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

CONF = 0.25

with open(ANNO) as f:
    gt = json.load(f)
fname_to_id = {Path(img['file_name']).name: img['id'] for img in gt['images']}
cat_ids     = sorted([c['id'] for c in gt['categories']])
yolo_to_cat = {idx: cid for idx, cid in enumerate(cat_ids)}
img_list    = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))

model = RTDETR(CKPT)

for imgsz in [480, 800]:
    print(f"\n=== 스케일 {imgsz} 추론 시작 ===")
    predictions = []

    for img_path in tqdm(img_list, desc=f"  imgsz={imgsz}"):
        fname = img_path.name
        if fname not in fname_to_id:
            continue
        img_id = fname_to_id[fname]

        results = model.predict(
            str(img_path),
            imgsz=imgsz,
            conf=CONF,
            device=0,
            verbose=False,
        )

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
                    "bbox":        [round(x1,2), round(y1,2),
                                   round(x2-x1,2), round(y2-y1,2)],
                    "score":       round(score, 4),
                })

    out_path = PRED_DIR / f"rtdetr_imgsz{imgsz}_predictions.json"
    with open(out_path, 'w') as f:
        json.dump(predictions, f)
    print(f"  저장: {out_path} ({len(predictions)}개)")

print("\n✅ TTA 추론 완료!")
