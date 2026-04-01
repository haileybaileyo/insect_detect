"""
03_predict_all.py
4개 모델 추론 스크립트
→ COCO 평가용 predictions.json + FPS 측정

실행 예시:
  python3 scripts/03_predict_all.py --model yolov8 --ckpt checkpoints/yolov8/yolov8n_imgsz640/weights/best.pt
  python3 scripts/03_predict_all.py --model rtdetr  --ckpt checkpoints/rtdetr/rtdetr_l_imgsz6409/weights/best.pt
  python3 scripts/03_predict_all.py --model faster_rcnn --ckpt checkpoints/faster_rcnn/faster_rcnn_imgsz640_best.pth
  python3 scripts/03_predict_all.py --model retinanet   --ckpt checkpoints/retinanet/retinanet_imgsz640_best.pth
"""

import argparse
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import cv2
from tqdm import tqdm

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
SEED     = 42
CONF     = 0.25
IOU_NMS  = 0.7
IMGSZ    = 640

BASE     = Path("/home/hailey/insect_detect_project/pest24_experiment")
ANNO_DIR = BASE / "data/annotations"
IMG_DIR  = BASE / "data/processed/test/images"
PRED_DIR = BASE / "results/predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────
def load_test_info():
    """COCO test JSON에서 이미지 정보 로드"""
    with open(ANNO_DIR / "instances_test.json") as f:
        coco = json.load(f)
    fname_to_id = {Path(img['file_name']).name: img['id'] for img in coco['images']}
    id_to_size  = {img['id']: (img['width'], img['height']) for img in coco['images']}
    cat_ids     = sorted([c['id'] for c in coco['categories']])
    return fname_to_id, id_to_size, cat_ids

def save_results(model_name, predictions, fps_times):
    """predictions.json + FPS 저장"""
    out_path = PRED_DIR / f"{model_name}_predictions.json"
    with open(out_path, 'w') as f:
        json.dump(predictions, f)

    fps_times = fps_times[5:]  # 워밍업 5장 제외
    if fps_times:
        avg_ms = np.mean(fps_times) * 1000
        fps    = 1.0 / np.mean(fps_times)
    else:
        avg_ms, fps = 0, 0

    fps_meta = {
        "model":           model_name,
        "num_predictions": len(predictions),
        "fps":             round(fps, 2),
        "ms_per_img":      round(avg_ms, 2),
    }
    fps_path = PRED_DIR / f"{model_name}_fps.json"
    with open(fps_path, 'w') as f:
        json.dump(fps_meta, f, indent=2)

    print(f"\n  FPS:      {fps:.1f} ({avg_ms:.1f} ms/img)")
    print(f"  예측 수:  {len(predictions)}")
    print(f"  저장:     {out_path}")
    print(f"\n✅ 추론 완료!")
    return fps


# ─────────────────────────────────────────────
# YOLO 계열 (YOLOv8, RT-DETR)
# ─────────────────────────────────────────────
def predict_yolo(model_type: str, ckpt: str):
    print("=" * 60)
    print(f" {model_type.upper()} 추론 시작")
    print(f" 체크포인트: {ckpt}")
    print("=" * 60)

    if model_type == "rtdetr":
        from ultralytics import RTDETR
        model = RTDETR(ckpt)
    else:
        from ultralytics import YOLO
        model = YOLO(ckpt)

    fname_to_id, id_to_size, cat_ids = load_test_info()
    yolo_to_cat = {idx: cid for idx, cid in enumerate(cat_ids)}

    img_list = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    print(f"  테스트 이미지: {len(img_list)}")

    predictions = []
    fps_times   = []
    device      = 0 if torch.cuda.is_available() else 'cpu'

    for img_path in tqdm(img_list, desc="  추론"):
        fname = img_path.name
        if fname not in fname_to_id:
            continue
        img_id = fname_to_id[fname]
        orig_w, orig_h = id_to_size[img_id]

        t0 = time.perf_counter()
        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ, conf=CONF, iou=IOU_NMS,
            verbose=False, device=device
        )
        t1 = time.perf_counter()
        fps_times.append(t1 - t0)

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                score  = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())

       

                predictions.append({
                    "image_id":    img_id,
                    "category_id": yolo_to_cat.get(cls_id, cls_id + 1),
                    "bbox":        [round(x1,2), round(y1,2),
                                    round(x2-x1,2), round(y2-y1,2)],
                    "score":       round(score, 4),
                })

    save_results(model_type, predictions, fps_times)


# ─────────────────────────────────────────────
# torchvision 계열 (Faster R-CNN, RetinaNet)
# ─────────────────────────────────────────────
def predict_torchvision(model_name: str, ckpt: str):
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        retinanet_resnet50_fpn,
    )

    print("=" * 60)
    print(f" {model_name.upper()} 추론 시작")
    print(f" 체크포인트: {ckpt}")
    print("=" * 60)

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt, map_location=device)
    num_classes = checkpoint.get('num_classes', 25)
    print(f"  Device: {device} | Classes: {num_classes}")

    # 모델 빌드
    if model_name == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feat, num_classes)
        )
    else:  # retinanet
        model = retinanet_resnet50_fpn(weights=None)
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.num_classes = num_classes
        model.head.classification_head.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )

    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    fname_to_id, id_to_size, cat_ids = load_test_info()
    label_to_cat = {idx+1: cid for idx, cid in enumerate(cat_ids)}

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMGSZ, IMGSZ)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    img_list = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    print(f"  테스트 이미지: {len(img_list)}")

    predictions = []
    fps_times   = []

    with torch.no_grad():
        for img_path in tqdm(img_list, desc="  추론"):
            fname = img_path.name
            if fname not in fname_to_id:
                continue
            img_id = fname_to_id[fname]
            orig_w, orig_h = id_to_size[img_id]

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = transform(image).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            output = model(tensor)[0]
            t1 = time.perf_counter()
            fps_times.append(t1 - t0)

            boxes  = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            # confidence threshold 적용
            mask   = scores >= CONF
            boxes  = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            scale_x = orig_w / IMGSZ
            scale_y = orig_h / IMGSZ

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                x1 *= scale_x; x2 *= scale_x
                y1 *= scale_y; y2 *= scale_y

                predictions.append({
                    "image_id":    img_id,
                    "category_id": label_to_cat.get(int(label), int(label)),
                    "bbox":        [round(float(x1),2), round(float(y1),2),
                                    round(float(x2-x1),2), round(float(y2-y1),2)],
                    "score":       round(float(score), 4),
                })

    save_results(model_name, predictions, fps_times)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["yolov8","rtdetr","faster_rcnn","retinanet"])
    parser.add_argument("--ckpt",  required=True, help="체크포인트 경로")
    args = parser.parse_args()

    if args.model in ["yolov8", "rtdetr"]:
        predict_yolo(args.model, args.ckpt)
    else:
        predict_torchvision(args.model, args.ckpt)