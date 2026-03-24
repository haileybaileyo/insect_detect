import os
import json
import shutil
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
SEED       = 42
DATA_ROOT  = Path("/home/hailey/insect_detect_project/pest24_raw")
OUT_ROOT   = Path("/home/hailey/insect_detect_project/pest24_experiment/data/processed")
ANNO_OUT   = Path("/home/hailey/insect_detect_project/pest24_experiment/data/annotations")
CFG_OUT    = Path("/home/hailey/insect_detect_project/pest24_experiment/configs")

random.seed(SEED)
np.random.seed(SEED)

# 클래스 목록 (알파벳 정렬 → 고정 인덱스)
CLASSES = sorted([
    "Agriotes fuscicollis Miwa", "Anomala corpulenta", "Armyworm",
    "Athetis lepigone", "Bollworm", "Gryllotalpa orientalis",
    "Land tiger", "Little Gecko", "Meadow borer", "Melahotus",
    "Nematode trench", "Plutella xylostella", "Rice Leaf Roller",
    "Rice planthopper", "Scotogramma trifolii Rottemberg",
    "Spodoptera cabbage", "Spodoptera exigua", "Spodoptera litura",
    "Stem borer", "Striped rice bore", "Yellow tiger",
    "eight-character tiger", "holotrichia oblita", "holotrichia parallela"
])
CLS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
CLS_TO_CID = {cls: idx+1 for idx, cls in enumerate(CLASSES)}  # COCO: 1-indexed

print("=" * 60)
print(" Pest24 데이터 분할 시작 (8:1:1, seed=42)")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. XML 파싱
# ─────────────────────────────────────────────
print("\n[1] XML 파싱 중...")

xml_files = sorted((DATA_ROOT / "Annotations").glob("*.xml"))
all_data = []  # (stem, dominant_class, objects)

for xml_file in tqdm(xml_files):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        objects = []
        cls_counts = Counter()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLS_TO_IDX:
                continue
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            # 클램핑
            xmin = max(0, min(img_w, xmin))
            ymin = max(0, min(img_h, ymin))
            xmax = max(0, min(img_w, xmax))
            ymax = max(0, min(img_h, ymax))
            if xmax <= xmin or ymax <= ymin:
                continue
            objects.append({
                "name": name,
                "xmin": xmin, "ymin": ymin,
                "xmax": xmax, "ymax": ymax,
                "img_w": img_w, "img_h": img_h
            })
            cls_counts[name] += 1

        if not objects:
            continue

        dominant = cls_counts.most_common(1)[0][0]
        all_data.append((xml_file.stem, dominant, objects, img_w, img_h))
    except Exception as e:
        print(f"  파싱 오류: {xml_file.name} — {e}")

print(f"  유효 이미지: {len(all_data)}")

# ─────────────────────────────────────────────
# 2. 8:1:1 분할
# ─────────────────────────────────────────────
print("\n[2] 8:1:1 분할 중...")

stems      = [d[0] for d in all_data]
dom_labels = [d[1] for d in all_data]

train_stems, temp_stems, _, temp_labels = train_test_split(
    stems, dom_labels, test_size=0.2, random_state=SEED, stratify=dom_labels
)
val_stems, test_stems = train_test_split(
    temp_stems, test_size=0.5, random_state=SEED, stratify=temp_labels
)

stem_to_data = {d[0]: d for d in all_data}

print(f"  Train: {len(train_stems)}")
print(f"  Val:   {len(val_stems)}")
print(f"  Test:  {len(test_stems)}")

# ─────────────────────────────────────────────
# 3. 이미지 복사 + YOLO txt + COCO JSON 저장
# ─────────────────────────────────────────────
def save_split(split_name, split_stems):
    img_out = OUT_ROOT / split_name / "images"
    lbl_out = OUT_ROOT / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_id, stem in enumerate(tqdm(split_stems, desc=f"  [{split_name}]"), 1):
        _, dominant, objects, img_w, img_h = stem_to_data[stem]

        # 이미지 복사
        src = DATA_ROOT / "images" / f"{stem}.jpg"
        if not src.exists():
            src = DATA_ROOT / "images" / f"{stem}.png"
        if src.exists():
            shutil.copy2(src, img_out / src.name)
            fname = src.name
        else:
            continue

        # COCO images
        coco_images.append({
            "id": img_id,
            "file_name": fname,
            "width": img_w,
            "height": img_h
        })

        # YOLO txt
        yolo_lines = []
        for obj in objects:
            cls_idx = CLS_TO_IDX[obj["name"]]
            xc = ((obj["xmin"] + obj["xmax"]) / 2) / img_w
            yc = ((obj["ymin"] + obj["ymax"]) / 2) / img_h
            nw = (obj["xmax"] - obj["xmin"]) / img_w
            nh = (obj["ymax"] - obj["ymin"]) / img_h
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            # COCO annotation
            bw = obj["xmax"] - obj["xmin"]
            bh = obj["ymax"] - obj["ymin"]
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLS_TO_CID[obj["name"]],
                "bbox": [round(obj["xmin"],2), round(obj["ymin"],2),
                         round(bw,2), round(bh,2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0
            })
            ann_id += 1

        with open(lbl_out / f"{stem}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

    # COCO JSON 저장
    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": CLS_TO_CID[cls], "name": cls} for cls in CLASSES]
    }
    with open(ANNO_OUT / f"instances_{split_name}.json", "w") as f:
        json.dump(coco_dict, f)

    print(f"    ✅ {split_name}: {len(coco_images)}장, {len(coco_annotations)}개 객체")
    return len(coco_images)

print("\n[3] 파일 저장 중...")
save_split("train", train_stems)
save_split("val",   val_stems)
save_split("test",  test_stems)

# ─────────────────────────────────────────────
# 4. data.yaml 저장
# ─────────────────────────────────────────────
import yaml
data_yaml = {
    "path": str(OUT_ROOT.resolve()),
    "train": "train/images",
    "val":   "val/images",
    "test":  "test/images",
    "nc":    len(CLASSES),
    "names": CLASSES
}
CFG_OUT.mkdir(parents=True, exist_ok=True)
with open(CFG_OUT / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)

print(f"\n  ✅ data.yaml 저장 완료")
print("\n✅ 데이터 분할 완료!")
print(f"   Train: {len(train_stems)} | Val: {len(val_stems)} | Test: {len(test_stems)}")