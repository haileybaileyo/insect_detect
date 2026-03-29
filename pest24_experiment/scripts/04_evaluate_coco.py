"""
04_evaluate_coco.py
4개 모델 전체 평가 스크립트
→ mAP, AP_small, Dense AP, Sparse AP, FPS, Small FN Rate, Class-wise mAP

실행:
  # 단일 모델 평가
  python3 scripts/04_evaluate_coco.py --model yolov8

  # 4개 모델 한번에 비교표 생성
  python3 scripts/04_evaluate_coco.py --compare
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
BASE       = Path("/home/hailey/insect_detect_project/pest24_experiment")
ANNO_FILE  = str(BASE / "data/annotations/instances_test.json")
PRED_DIR   = BASE / "results/predictions"
METRIC_DIR = BASE / "results/metrics"
METRIC_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 단일 모델 평가
# ─────────────────────────────────────────────
def evaluate_one(model_name: str) -> dict:
    pred_file = PRED_DIR / f"{model_name}_predictions.json"
    if not pred_file.exists():
        print(f"❌ {pred_file} 없음 — 먼저 03_predict_all.py 실행하세요")
        return {}

    print("=" * 60)
    print(f" 평가: {model_name}")
    print("=" * 60)

    coco_gt  = COCO(ANNO_FILE)
    coco_dt  = coco_gt.loadRes(str(pred_file))

    # ── 기본 COCO 평가
    eval_obj = COCOeval(coco_gt, coco_dt, iouType='bbox')
    eval_obj.evaluate()
    eval_obj.accumulate()
    eval_obj.summarize()

    stats = eval_obj.stats
    metrics = {
        "model":          model_name,
        "mAP_0.5:0.95":   round(float(stats[0]), 4),
        "mAP_0.5":        round(float(stats[1]), 4),
        "mAP_0.75":       round(float(stats[2]), 4),
        "AP_small":       round(float(stats[3]), 4),
        "AP_medium":      round(float(stats[4]), 4),
        "AP_large":       round(float(stats[5]), 4),
        "AR_1":           round(float(stats[6]), 4),
        "AR_10":          round(float(stats[7]), 4),
        "AR_100":         round(float(stats[8]), 4),
        "AR_small":       round(float(stats[9]), 4),
        "AR_medium":      round(float(stats[10]), 4),
        "AR_large":       round(float(stats[11]), 4),
    }

    # ── FPS 로드
    fps_file = PRED_DIR / f"{model_name}_fps.json"
    if fps_file.exists():
        with open(fps_file) as f:
            fps_data = json.load(f)
        metrics["fps"]        = fps_data.get("fps")
        metrics["ms_per_img"] = fps_data.get("ms_per_img")
    else:
        metrics["fps"]        = None
        metrics["ms_per_img"] = None

    # ── Dense / Sparse AP
    density = _density_ap(coco_gt, coco_dt)
    metrics.update(density)

    # ── Class-wise mAP
    metrics["classwise_mAP"] = _classwise_map(coco_gt, coco_dt)

    # ── Small FN Rate
    metrics["small_FN_rate"] = _small_fn_rate(coco_gt, pred_file)

    # 저장
    out = METRIC_DIR / f"{model_name}_metrics.json"
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _print_summary(metrics)
    print(f"\n✅ 지표 저장: {out}")
    return metrics


# ─────────────────────────────────────────────
# Dense / Sparse AP
# ─────────────────────────────────────────────
def _density_ap(coco_gt, coco_dt) -> dict:
    """이미지당 GT 객체 수 기준으로 Dense/Sparse 분리 평가"""
    ann_by_img = defaultdict(list)
    for ann in coco_gt.dataset['annotations']:
        ann_by_img[ann['image_id']].append(ann)

    sparse_ids = [iid for iid, anns in ann_by_img.items() if 1  <= len(anns) <= 9]
    dense_ids  = [iid for iid, anns in ann_by_img.items() if len(anns) >= 10]

    def _ap_subset(img_ids):
        if not img_ids:
            return -1.0
        ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return round(float(ev.stats[1]), 4)  # mAP@0.5

    sparse_ap = _ap_subset(sparse_ids)
    dense_ap  = _ap_subset(dense_ids)

    print(f"\n  Density 분석:")
    print(f"    Sparse (1~9  obj/img, n={len(sparse_ids)}): AP@0.5 = {sparse_ap:.4f}")
    print(f"    Dense  (≥10 obj/img, n={len(dense_ids)}):  AP@0.5 = {dense_ap:.4f}")
    print(f"    갭 (Sparse - Dense): {sparse_ap - dense_ap:.4f}")

    return {
        "AP_sparse":    sparse_ap,
        "AP_dense":     dense_ap,
        "density_gap":  round(sparse_ap - dense_ap, 4),
        "n_sparse":     len(sparse_ids),
        "n_dense":      len(dense_ids),
    }


# ─────────────────────────────────────────────
# Class-wise mAP
# ─────────────────────────────────────────────
def _classwise_map(coco_gt, coco_dt) -> dict:
    """카테고리별 AP@0.5"""
    cat_ap = {}
    for cat in coco_gt.dataset['categories']:
        cat_id   = cat['id']
        cat_name = cat['name']
        ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
        ev.params.catIds = [cat_id]
        try:
            ev.evaluate(); ev.accumulate(); ev.summarize()
            ap = round(float(ev.stats[1]), 4)
        except:
            ap = -1.0
        cat_ap[cat_name] = ap

    # 하위 5개 (어려운 클래스)
    sorted_ap = sorted(cat_ap.items(), key=lambda x: x[1])
    print(f"\n  Class-wise mAP (하위 5개 — 어려운 클래스):")
    for name, ap in sorted_ap[:5]:
        print(f"    {name}: {ap:.4f}")
    print(f"\n  Class-wise mAP (상위 5개 — 쉬운 클래스):")
    for name, ap in sorted_ap[-5:]:
        print(f"    {name}: {ap:.4f}")

    return cat_ap


# ─────────────────────────────────────────────
# Small FN Rate
# ─────────────────────────────────────────────
def _small_fn_rate(coco_gt, pred_file) -> float:
    """area < 32² GT 중 탐지 못한 비율"""
    with open(pred_file) as f:
        preds = json.load(f)

    pred_by_img = defaultdict(list)
    for p in preds:
        pred_by_img[p['image_id']].append(p)

    total_small  = 0
    missed_small = 0

    for ann in coco_gt.dataset['annotations']:
        x, y, w, h = ann['bbox']
        area = ann.get('area', w * h)
        if area >= 32 * 32:
            continue
        total_small += 1
        img_id = ann['image_id']
        gx1, gy1, gx2, gy2 = x, y, x+w, y+h

        matched = False
        for p in pred_by_img[img_id]:
            if p['category_id'] != ann['category_id']:
                continue
            px, py, pw, ph = p['bbox']
            px1, py1, px2, py2 = px, py, px+pw, py+ph
            xi1 = max(gx1,px1); yi1 = max(gy1,py1)
            xi2 = min(gx2,px2); yi2 = min(gy2,py2)
            inter = max(0,xi2-xi1) * max(0,yi2-yi1)
            union = (gx2-gx1)*(gy2-gy1) + (px2-px1)*(py2-py1) - inter
            iou = inter/union if union > 0 else 0
            if iou >= 0.5:
                matched = True
                break
        if not matched:
            missed_small += 1

    fn_rate = missed_small / total_small if total_small > 0 else 0.0
    print(f"\n  Small FN Rate: {missed_small}/{total_small} = {fn_rate*100:.1f}%")
    return round(fn_rate, 4)


# ─────────────────────────────────────────────
# 요약 출력
# ─────────────────────────────────────────────
def _print_summary(m: dict):
    print(f"\n  {'='*45}")
    print(f"  {m['model']} 평가 결과")
    print(f"  {'='*45}")
    print(f"  mAP@0.5:0.95  : {m['mAP_0.5:0.95']:.4f}")
    print(f"  mAP@0.5       : {m['mAP_0.5']:.4f}")
    print(f"  AP_small      : {m['AP_small']:.4f}")
    print(f"  AR_small      : {m['AR_small']:.4f}")
    print(f"  AP_medium     : {m['AP_medium']:.4f}")
    print(f"  FPS           : {m.get('fps', 'N/A')}")
    print(f"  Dense AP      : {m.get('AP_dense', 'N/A')}")
    print(f"  Sparse AP     : {m.get('AP_sparse', 'N/A')}")
    print(f"  Density 갭    : {m.get('density_gap', 'N/A')}")
    print(f"  Small FN Rate : {m.get('small_FN_rate', 'N/A')}")


# ─────────────────────────────────────────────
# 4개 모델 비교표
# ─────────────────────────────────────────────
def compare_all():
    metric_files = sorted(METRIC_DIR.glob("*_metrics.json"))
    if not metric_files:
        print("❌ 평가된 모델 없음 — 먼저 개별 평가 실행하세요")
        return

    rows = []
    for mf in metric_files:
        with open(mf) as f:
            m = json.load(f)
        rows.append({
            "Model":         m.get("model"),
            "mAP@0.5:0.95":  m.get("mAP_0.5:0.95"),
            "mAP@0.5":       m.get("mAP_0.5"),
            "AP_small":      m.get("AP_small"),
            "AR_small":      m.get("AR_small"),
            "AP_medium":     m.get("AP_medium"),
            "FPS":           m.get("fps"),
            "Dense AP":      m.get("AP_dense"),
            "Sparse AP":     m.get("AP_sparse"),
            "Density 갭":    m.get("density_gap"),
            "Small FN Rate": m.get("small_FN_rate"),
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(" 4개 모델 성능 비교표")
    print("=" * 100)
    print(df.to_string(index=False))

    # CSV 저장
    csv_path = METRIC_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 비교표 저장: {csv_path}")

    # LaTeX 저장 (논문용)
    latex_path = METRIC_DIR / "comparison_table.tex"
    df.to_latex(latex_path, index=False, float_format="%.4f")
    print(f"✅ LaTeX 저장:  {latex_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default=None,
                        choices=["yolov8","rtdetr","faster_rcnn","retinanet"])
    parser.add_argument("--compare", action="store_true",
                        help="4개 모델 비교표 생성")
    args = parser.parse_args()

    if args.compare:
        compare_all()
    elif args.model:
        evaluate_one(args.model)
    else:
        # 기본: 4개 모델 순서대로 전부 평가 후 비교표
        print("4개 모델 전체 평가 시작...")
        for model in ["yolov8", "rtdetr", "faster_rcnn", "retinanet"]:
            pred_file = PRED_DIR / f"{model}_predictions.json"
            if pred_file.exists():
                evaluate_one(model)
            else:
                print(f"⚠️  {model} 예측 파일 없음 — 스킵")
        compare_all()