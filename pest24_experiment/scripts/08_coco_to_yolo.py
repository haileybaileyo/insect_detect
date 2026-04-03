"""
08_coco_to_yolo.py
COCO JSON → YOLO txt 변환
슬라이스 데이터셋용
"""
import json
import os
from pathlib import Path
from tqdm import tqdm

COCO_JSON = "pest24_experiment/data/sliced_320/instances_train_sliced_coco.json"
OUTPUT_DIR = "pest24_experiment/data/sliced_320/labels"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

with open(COCO_JSON) as f:
    coco = json.load(f)

# image_id → (file_name, width, height)
id_to_img = {img['id']: img for img in coco['images']}

# category_id → 0-indexed class
cat_ids = sorted([c['id'] for c in coco['categories']])
cat_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}

# image_id → annotations
from collections import defaultdict
img_to_anns = defaultdict(list)
for ann in coco['annotations']:
    img_to_anns[ann['image_id']].append(ann)

print(f"변환 시작: {len(coco['images'])}장")

for img_info in tqdm(coco['images']):
    img_id = img_info['id']
    w, h = img_info['width'], img_info['height']
    stem = Path(img_info['file_name']).stem

    lines = []
    for ann in img_to_anns[img_id]:
        x, y, bw, bh = ann['bbox']
        # COCO [x,y,w,h] → YOLO [cx,cy,w,h] normalized
        cx = (x + bw/2) / w
        cy = (y + bh/2) / h
        nw = bw / w
        nh = bh / h
        cls = cat_to_idx[ann['category_id']]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    with open(f"{OUTPUT_DIR}/{stem}.txt", 'w') as f:
        f.write('\n'.join(lines))

print(f"✅ 완료! 라벨 저장: {OUTPUT_DIR}")
