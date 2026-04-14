
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

구간 = ["0~8px", "8~16px", "16~24px", "24~32px", "32px+"]
x = np.arange(len(구간))

data = {
    "RT-DETR":      [14.8, 66.7, 90.0, 94.1, 95.7],
    "Faster R-CNN": [7.4,  55.6, 80.8, 89.5, 94.5],
    "YOLOv8":       [3.7,  22.0, 68.1, 85.9, 91.6],
    "RetinaNet":    [3.7,  26.0, 55.2, 79.8, 91.5],
}
colors  = ["#2E5FA3", "#E07B39", "#3A9E5F", "#C0392B"]
markers = ["o", "s", "^", "D"]
lines   = ["-", "--", "-.", ":"]

fig, ax = plt.subplots(figsize=(6.5, 3.8))

for (model, vals), color, marker, ls in zip(data.items(), colors, markers, lines):
    ax.plot(x, vals, color=color, marker=marker, linestyle=ls,
            linewidth=1.8, markersize=6, label=model)

ax.axvspan(0.5, 1.5, alpha=0.08, color="red")
ax.axvline(x=1, color="red", linestyle="--", linewidth=1.0, alpha=0.5)
ax.text(1.02, 3, "16px 임계점", color="red", fontsize=8, ha="left", va="bottom", alpha=0.85)

ax.annotate("", xy=(1, 66.7), xytext=(1, 90.0),
            arrowprops=dict(arrowstyle="<->", color="#2E5FA3", lw=1.4))
ax.text(1.12, 77, "23.3%p", color="#2E5FA3", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(구간, fontsize=9)
ax.set_ylabel("탐지율 (%)", fontsize=10)
ax.set_xlabel("객체 크기 구간 (변의 길이 기준)", fontsize=10)
ax.set_ylim(0, 108)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.set_tick_params(labelsize=9)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("threshold_16px.png", dpi=220, bbox_inches="tight")
print("저장 완료: threshold_16px.png")
