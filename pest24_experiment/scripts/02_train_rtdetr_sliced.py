"""
02_train_rtdetr_sliced.py
RT-DETR + SF (Slicing Fine-tuning)
슬라이스 데이터로 재학습
"""
from ultralytics import RTDETR
import torch

print("=" * 60)
print(" RT-DETR Slicing Fine-tuning 시작")
print(" 데이터: sliced_320 (182,718장)")
print("=" * 60)

# 기존 best.pt에서 파인튜닝
model = RTDETR("rtdetr-l.pt")

model.train(
    data="pest24_experiment/configs/data_sliced_320.yaml",
    epochs=100,
    imgsz=320,
    batch=16,
    optimizer="AdamW",
    lr0=0.00005,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    warmup_epochs=5,
    cos_lr=True,
    patience=15,
    save=True,
    project="pest24_experiment/checkpoints/rtdetr_sliced",
    name="rtdetr_l_sliced_320",
    seed=42,
    device=0,
    workers=4,
    verbose=True,
)

print("✅ RT-DETR Slicing Fine-tuning 완료!")
