"""
02_train_faster_rcnn.py
Faster R-CNN 학습 스크립트 — 공식 권장 설정 (방향 B)
torchvision 기반, COCO JSON 포맷 사용
"""

import os
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm

# ─────────────────────────────────────────────
# 설정 (공식 권장값)
# ─────────────────────────────────────────────
SEED         = 42
EPOCHS       = 100
PATIENCE     = 20
LR           = 0.02
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0001
BATCH        = 2
IMGSZ        = 640
CONF         = 0.25

ANNO_DIR  = Path("/home/hailey/insect_detect_project/pest24_experiment/data/annotations")
IMG_ROOT  = Path("/home/hailey/insect_detect_project/pest24_experiment/data/processed")
CKPT_DIR  = Path("/home/hailey/insect_detect_project/pest24_experiment/checkpoints/faster_rcnn")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────
class PestDataset(Dataset):
    def __init__(self, anno_file, img_dir, imgsz, is_train=True):
        with open(anno_file) as f:
            coco = json.load(f)

        self.img_dir  = Path(img_dir)
        self.imgsz    = imgsz
        self.is_train = is_train

        self.id_to_img  = {img['id']: img for img in coco['images']}
        self.ann_by_img = defaultdict(list)
        for ann in coco['annotations']:
            self.ann_by_img[ann['image_id']].append(ann)

        self.img_ids = [iid for iid in self.id_to_img if self.ann_by_img[iid]]

        cat_ids = sorted([c['id'] for c in coco['categories']])
        self.cat_to_label = {cid: idx+1 for idx, cid in enumerate(cat_ids)}
        self.num_classes  = len(cat_ids) + 1

        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
                A.Resize(imgsz, imgsz),
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1))
        else:
            self.transform = A.Compose([
                A.Resize(imgsz, imgsz),
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.id_to_img[img_id]
        img_path = self.img_dir / Path(img_info['file_name']).name

        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        bboxes, labels = [], []
        for ann in self.ann_by_img[img_id]:
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x+w, y+h
            x1 = max(0, min(orig_w-1, x1))
            y1 = max(0, min(orig_h-1, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            bboxes.append([x1, y1, x2, y2])
            labels.append(self.cat_to_label[ann['category_id']])

        if not bboxes:
            bboxes = [[0, 0, 1, 1]]
            labels = [0]

        try:
            t = self.transform(image=image, bboxes=bboxes, labels=labels)
            image  = t['image']
            bboxes = t['bboxes']
            labels = t['labels']
        except:
            image  = torch.zeros(3, self.imgsz, self.imgsz)
            bboxes = [[0, 0, 1, 1]]
            labels = [0]

        if not bboxes:
            bboxes = [[0, 0, 1, 1]]
            labels = [0]

        target = {
            'boxes':    torch.tensor(bboxes, dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ─────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0)}")

    run_name = f"faster_rcnn_imgsz{IMGSZ}"

    train_ds = PestDataset(str(ANNO_DIR/"instances_train.json"), str(IMG_ROOT/"train/images"), IMGSZ, True)
    val_ds   = PestDataset(str(ANNO_DIR/"instances_val.json"),   str(IMG_ROOT/"val/images"),   IMGSZ, False)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {train_ds.num_classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # 모델
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, train_ds.num_classes)
    model.to(device)

    # 공식 권장: SGD + StepLR
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_loss = float('inf')
    patience_counter = 0
    history = []

    print("=" * 60)
    print(f" Faster R-CNN 학습 시작 | imgsz={IMGSZ} | epochs={EPOCHS}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, targets in tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS}", leave=False):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0

        # Val loss
        model.train()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images  = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss_dict.values()).item()
        val_loss /= len(val_loader)

        print(f"  Epoch {epoch:3d} | Train Loss={avg_loss:.4f} | Val Loss={val_loss:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss})

        # Best 저장
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            ckpt_path = CKPT_DIR / f"{run_name}_best.pth"
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_loss, "num_classes": train_ds.num_classes,
                "imgsz": IMGSZ,
            }, ckpt_path)
            print(f"  ✅ Best 저장 (val_loss={best_loss:.4f}): {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  ⏹️  Early Stop! (best epoch 기준 patience {PATIENCE} 초과)")
                break

    # 히스토리 저장
    with open(CKPT_DIR / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Faster R-CNN 학습 완료!")
    print(f"   best 모델: {CKPT_DIR / f'{run_name}_best.pth'}")
    print(f"   best val_loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()