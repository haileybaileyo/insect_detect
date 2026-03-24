"""
02_train_rtdetr.py
RT-DETR 학습 스크립트 — 공식 권장 설정 (방향 B)
"""

from ultralytics import RTDETR
import torch
import json
from pathlib import Path

# ─────────────────────────────────────────────
# 설정 (공식 권장값)
# ─────────────────────────────────────────────
MODEL_NAME = "rtdetr-l.pt"
DATA_YAML  = "/home/hailey/insect_detect_project/pest24_experiment/configs/data.yaml"
CKPT_DIR   = "/home/hailey/insect_detect_project/pest24_experiment/checkpoints/rtdetr"
IMGSZ      = 640
EPOCHS     = 100
PATIENCE   = 20
BATCH      = 8
SEED       = 42

# RT-DETR 공식 권장값
LR0          = 0.0001
LRF          = 0.0001
WEIGHT_DECAY = 0.0001

print("=" * 60)
print(f" RT-DETR 학습 시작")
print(f" 모델: {MODEL_NAME} | imgsz: {IMGSZ} | batch: {BATCH}")
print(f" GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

model = RTDETR(MODEL_NAME)

results = model.train(
    data         = DATA_YAML,
    epochs       = EPOCHS,
    patience     = PATIENCE,
    imgsz        = IMGSZ,
    batch        = BATCH,
    seed         = SEED,
    # 공식 권장 optimizer 설정
    optimizer    = "AdamW",
    lr0          = LR0,
    lrf          = LRF,
    weight_decay = WEIGHT_DECAY,
    # 증강 (공통)
    fliplr       = 0.5,
    flipud       = 0.0,
    degrees      = 10.0,
    hsv_v        = 0.4,
    # 저장
    project      = CKPT_DIR,
    name         = f"rtdetr_l_imgsz{IMGSZ}",
    save         = True,
    save_period  = 10,
    # 평가
    conf         = 0.25,
    iou          = 0.7,
    plots        = True,
    verbose      = True,
    device       = 0,
)

# 결과 저장
best = Path(CKPT_DIR) / f"rtdetr_l_imgsz{IMGSZ}" / "weights" / "best.pt"
meta = {
    "model":      MODEL_NAME,
    "imgsz":      IMGSZ,
    "batch":      BATCH,
    "optimizer":  "AdamW",
    "lr0":        LR0,
    "best_ckpt":  str(best),
    "mAP50":      results.results_dict.get("metrics/mAP50(B)"),
    "mAP50_95":   results.results_dict.get("metrics/mAP50-95(B)"),
}
with open(Path(CKPT_DIR) / f"rtdetr_l_imgsz{IMGSZ}" / "train_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ 학습 완료!")
print(f"   best 모델: {best}")
print(f"   mAP@0.5:     {meta['mAP50']}")
print(f"   mAP@0.5:0.95:{meta['mAP50_95']}")