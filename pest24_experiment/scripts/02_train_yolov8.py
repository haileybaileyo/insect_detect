"""
02_train_yolov8.py
YOLOv8 학습 스크립트 — 공식 권장 설정 (방향 B)
"""

from ultralytics import YOLO
import torch
import json
from pathlib import Path

# ─────────────────────────────────────────────
# 설정 (공식 권장값)
# ─────────────────────────────────────────────
MODEL_NAME = "yolov8n.pt"
DATA_YAML  = "/home/hailey/insect_detect_project/pest24_experiment/configs/data.yaml"
CKPT_DIR   = "/home/hailey/insect_detect_project/pest24_experiment/checkpoints/yolov8"
IMGSZ      = 640
EPOCHS     = 100
PATIENCE   = 20
BATCH      = 16
SEED       = 42

# YOLOv8 공식 권장값
LR0        = 0.01
LRF        = 0.01
MOMENTUM   = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0

print("=" * 60)
print(f" YOLOv8 학습 시작")
print(f" 모델: {MODEL_NAME} | imgsz: {IMGSZ} | batch: {BATCH}")
print(f" GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

model = YOLO(MODEL_NAME)

results = model.train(
    data          = DATA_YAML,
    epochs        = EPOCHS,
    patience      = PATIENCE,
    imgsz         = IMGSZ,
    batch         = BATCH,
    seed          = SEED,
    # 공식 권장 optimizer 설정
    optimizer     = "SGD",
    lr0           = LR0,
    lrf           = LRF,
    momentum      = MOMENTUM,
    weight_decay  = WEIGHT_DECAY,
    warmup_epochs = WARMUP_EPOCHS,
    # 증강 (공통)
    fliplr        = 0.5,
    flipud        = 0.0,
    degrees       = 10.0,
    hsv_v         = 0.4,
    mosaic        = 1.0,
    # 저장
    project       = CKPT_DIR,
    name          = f"yolov8n_imgsz{IMGSZ}",
    save          = True,
    save_period   = 10,
    # 평가
    conf          = 0.25,
    iou           = 0.7,
    plots         = True,
    verbose       = True,
    device        = 0,
)

# 결과 저장
best = Path(CKPT_DIR) / f"yolov8n_imgsz{IMGSZ}" / "weights" / "best.pt"
meta = {
    "model":      MODEL_NAME,
    "imgsz":      IMGSZ,
    "batch":      BATCH,
    "optimizer":  "SGD",
    "lr0":        LR0,
    "best_ckpt":  str(best),
    "mAP50":      results.results_dict.get("metrics/mAP50(B)"),
    "mAP50_95":   results.results_dict.get("metrics/mAP50-95(B)"),
}
with open(Path(CKPT_DIR) / f"yolov8n_imgsz{IMGSZ}" / "train_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ 학습 완료!")
print(f"   best 모델: {best}")
print(f"   mAP@0.5:     {meta['mAP50']}")
print(f"   mAP@0.5:0.95:{meta['mAP50_95']}")