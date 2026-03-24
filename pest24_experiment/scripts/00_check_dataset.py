import os
from pathlib import Path
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET

DATA_ROOT = Path("/home/hailey/insect_detect_project/pest24_raw")

print("=" * 60)
print(" Pest24 데이터셋 구조 분석")
print("=" * 60)

# 1. 이미지 수 확인
images = list((DATA_ROOT / "images").glob("*"))
print(f"\n[1] 총 이미지 수: {len(images)}")
print(f"    샘플: {images[0].name}")

# 2. XML 파싱 → 클래스, bbox 크기, 객체 수 분석
print("\n[2] Annotations 분석 중... (시간 조금 걸려요)")

class_counter = Counter()
area_small = 0
area_medium = 0
area_large = 0
total_objects = 0
obj_per_img = []

xml_files = list((DATA_ROOT / "Annotations").glob("*.xml"))

for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        count = 0
        for obj in root.findall("object"):
            name = obj.find("name").text
            class_counter[name] += 1
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            area = (xmax - xmin) * (ymax - ymin)
            if area < 32*32:
                area_small += 1
            elif area < 96*96:
                area_medium += 1
            else:
                area_large += 1
            total_objects += 1
            count += 1
        obj_per_img.append(count)
    except:
        pass

print(f"\n[3] 클래스 수: {len(class_counter)}")
print(f"    클래스 목록:")
for cls, cnt in sorted(class_counter.items()):
    print(f"      {cls}: {cnt}개")

print(f"\n[4] 총 객체 수: {total_objects}")
print(f"    Small  (area<32²):  {area_small} ({area_small/total_objects*100:.1f}%)")
print(f"    Medium (32²~96²):   {area_medium} ({area_medium/total_objects*100:.1f}%)")
print(f"    Large  (area≥96²):  {area_large} ({area_large/total_objects*100:.1f}%)")

print(f"\n[5] 이미지당 객체 수")
print(f"    평균: {sum(obj_per_img)/len(obj_per_img):.1f}")
print(f"    최소: {min(obj_per_img)}, 최대: {max(obj_per_img)}")
dense = sum(1 for c in obj_per_img if c >= 10)
print(f"    밀집(≥10 obj) 이미지: {dense}/{len(obj_per_img)} ({dense/len(obj_per_img)*100:.1f}%)")

print("\n✅ 데이터셋 분석 완료!")