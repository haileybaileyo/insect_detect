"""
13_train_rtdetr_896.py
RT-DETR 896 해상도 학습
소형 객체 탐지 개선 목적
"""
from ultralytics import RTDETR

print("=" * 60)
print(" RT-DETR 896 학습 시작")
print(" 목적: 입력 해상도 증가 → AP_small 개선")
print("=" * 60)

model = RTDETR("rtdetr-l.pt")

model.train(
    data="pest24_experiment/configs/data.yaml",
    epochs=100,
    imgsz=896,
    batch=8,
    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    warmup_epochs=5,
    cos_lr=True,
    patience=20,
    save=True,
    project="pest24_experiment/checkpoints/rtdetr_896",
    name="rtdetr_l_imgsz896",
    seed=42,
    device=0,
    workers=4,
    verbose=True,
)

print("✅ RT-DETR 896 학습 완료!")
