"""
09_slice_val.py
val 데이터 320×320 슬라이싱
"""
import os
import gc
from pathlib import Path
from sahi.slicing import slice_coco

OUTPUT_DIR = "pest24_experiment/data/sliced_320_val/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Val 슬라이싱 시작...")
result = slice_coco(
    coco_annotation_file_path="pest24_experiment/data/annotations/instances_val.json",
    image_dir="pest24_experiment/data/processed/val/images/",
    output_coco_annotation_file_name="instances_val_sliced",
    output_dir=OUTPUT_DIR,
    slice_height=320,
    slice_width=320,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=False,
)
gc.collect()

total = len(list(Path(OUTPUT_DIR).glob("*.png")))
print(f"완료! 총 슬라이스: {total}개")
print(result)
