"""
05_plot_pr_curve.py
4개 모델 PR 곡선 + 주요 시각화
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ─────────────────────────────────────────────
BASE      = Path("/home/hailey/insect_detect_project/pest24_experiment")
ANNO_FILE = str(BASE / "data/annotations/instances_test.json")
PRED_DIR  = BASE / "results/predictions"
VIS_DIR   = BASE / "results/visuals"
VIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "RT-DETR":      ("rtdetr_predictions.json",      "#1D9E75", "-"),
    "Faster R-CNN": ("faster_rcnn_predictions.json", "#378ADD", "--"),
    "YOLOv8":       ("yolov8_predictions.json",      "#EF9F27", "-."),
    "RetinaNet":    ("retinanet_predictions.json",   "#D85A30", ":"),
}

# ─────────────────────────────────────────────
# 1. PR 곡선 (전체 mAP@0.5 기준)
# ─────────────────────────────────────────────
def get_pr_curve(pred_file):
    coco_gt = COCO(ANNO_FILE)
    coco_dt = coco_gt.loadRes(str(pred_file))
    ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
    ev.params.iouThrs = np.array([0.5])
    ev.evaluate()
    ev.accumulate()

    # precision/recall 추출 (IoU=0.5, area=all, maxDets=100)
    # shape: [T, R, K, A, M] = [1, 101, 24, 4, 3]
    prec = ev.eval['precision'][0, :, :, 0, 2]  # IoU=0.5, area=all, maxDets=100
    prec_mean = np.mean(prec, axis=1)  # 클래스 평균
    recall = np.linspace(0, 1, 101)
    ap = float(ev.stats[1])
    return recall, prec_mean, ap

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Pest24 Detection Results', fontsize=14, fontweight='bold', y=1.02)

# ── PR 곡선 ──
ax1 = axes[0]
ax1.set_title('PR Curve (mAP@0.5)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Recall', fontsize=11)
ax1.set_ylabel('Precision', fontsize=11)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

for name, (fname, color, ls) in MODELS.items():
    pred_file = PRED_DIR / fname
    if not pred_file.exists():
        print(f"  ⚠️ {fname} 없음 — 스킵")
        continue
    try:
        recall, precision, ap = get_pr_curve(pred_file)
        ax1.plot(recall, precision, color=color, linestyle=ls,
                 linewidth=2.5, label=f"{name} (AP={ap:.4f})")
        print(f"  {name}: AP@0.5 = {ap:.4f}")
    except Exception as e:
        print(f"  {name} 오류: {e}")

ax1.legend(loc='upper right', fontsize=10)

# ── FPS vs AP_small 산점도 ──
ax2 = axes[1]
ax2.set_title('FPS vs AP_small Trade-off', fontsize=12, fontweight='bold')
ax2.set_xlabel('FPS (↑ 빠름)', fontsize=11)
ax2.set_ylabel('AP_small (↑ 정확)', fontsize=11)
ax2.grid(True, alpha=0.3)

metrics = {
    "RT-DETR":      (46.6,  0.2942, "#1D9E75"),
    "Faster R-CNN": (62.9,  0.2691, "#378ADD"),
    "YOLOv8":       (136.9, 0.2251, "#EF9F27"),
    "RetinaNet":    (66.2,  0.2264, "#D85A30"),
}

for name, (fps, ap_small, color) in metrics.items():
    ax2.scatter(fps, ap_small, s=200, color=color, zorder=5)
    offset_x = -12 if name == "RetinaNet" else 3
    offset_y = 0.003 if name not in ["YOLOv8"] else -0.007
    ax2.annotate(name, (fps, ap_small),
                 xytext=(fps + offset_x, ap_small + offset_y),
                 fontsize=10, fontweight='bold', color=color)

# 이상적인 영역 표시
ax2.axhline(y=0.30, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(x=50,   color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(52, 0.31, '목표 영역', fontsize=9, color='gray')
ax2.fill_between([50, 160], [0.30, 0.30], [0.40, 0.40],
                  alpha=0.08, color='green', label='이상적 영역')
ax2.legend(fontsize=9)
ax2.set_xlim(0, 160)
ax2.set_ylim(0.18, 0.36)

plt.tight_layout()
out = VIS_DIR / "01_pr_curve_fps_apsmall.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ 저장: {out}")

# ─────────────────────────────────────────────
# 2. 모델 비교 막대그래프
# ─────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 6))
ax.set_title('4개 모델 성능 비교', fontsize=13, fontweight='bold')

model_names = ["RT-DETR", "Faster R-CNN", "YOLOv8", "RetinaNet"]
colors_list = ["#1D9E75", "#378ADD", "#EF9F27", "#D85A30"]

metrics_data = {
    "mAP@0.5":       [0.6931, 0.6649, 0.5807, 0.5762],
    "mAP@0.5:0.95":  [0.4164, 0.4033, 0.3298, 0.3531],
    "AP_small":      [0.2942, 0.2691, 0.2251, 0.2264],
    "Dense AP":      [0.6456, 0.6224, 0.5011, 0.4914],
}

x = np.arange(len(model_names))
width = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]
metric_colors = ["#2C2C2A", "#444441", "#888780", "#B4B2A9"]
metric_names = list(metrics_data.keys())

for i, (metric, vals) in enumerate(metrics_data.items()):
    bars = ax.bar(x + offsets[i]*width, vals, width,
                  label=metric, color=metric_colors[i], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 0.85)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out2 = VIS_DIR / "02_model_comparison_bar.png"
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: {out2}")

# ─────────────────────────────────────────────
# 3. Small FN Rate + Density 갭 비교
# ─────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Pest24 핵심 지표 비교', fontsize=13, fontweight='bold')

model_names_short = ["RT-DETR", "Faster\nR-CNN", "YOLOv8", "RetinaNet"]
bar_colors = ["#1D9E75", "#378ADD", "#EF9F27", "#D85A30"]

# Small FN Rate
ax3a = axes3[0]
fn_rates = [0.0848, 0.1485, 0.2258, 0.3049]
bars = ax3a.bar(model_names_short, fn_rates, color=bar_colors, alpha=0.85)
ax3a.set_title('Small FN Rate (낮을수록 좋음)', fontsize=11, fontweight='bold')
ax3a.set_ylabel('FN Rate', fontsize=10)
ax3a.set_ylim(0, 0.40)
for bar, val in zip(bars, fn_rates):
    ax3a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
              f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
ax3a.grid(axis='y', alpha=0.3)

# Density 갭
ax3b = axes3[1]
density_gaps = [0.0312, 0.0252, 0.0707, 0.0907]
bars = ax3b.bar(model_names_short, density_gaps, color=bar_colors, alpha=0.85)
ax3b.set_title('Density 갭 (낮을수록 밀집 환경 강함)', fontsize=11, fontweight='bold')
ax3b.set_ylabel('Sparse AP - Dense AP', fontsize=10)
ax3b.set_ylim(0, 0.12)
for bar, val in zip(bars, density_gaps):
    ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
              f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax3b.grid(axis='y', alpha=0.3)

plt.tight_layout()
out3 = VIS_DIR / "03_fn_rate_density_gap.png"
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: {out3}")

print("\n🎉 모든 시각화 완료!")
print(f"저장 위치: {VIS_DIR}")
