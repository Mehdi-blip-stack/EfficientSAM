"""Experiments D and E: held-out mIoU against the number of labelled fine-tuning images (100 / 200 / 800 / 5,000)."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
T = json.load(open(sys.argv[1])); out = sys.argv[2]
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 220, "savefig.bbox": "tight"})
N = [100, 200, 800, 5000]; SUF = {100: "_n100", 200: "_n200", 800: "", 5000: "_5k"}
INIT = {"sami": ("SAMI start", "#e34948", "o"), "mae": ("MAE start", "#eda100", "s"), "random": ("Random start", "#8a8986", "^")}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, mode, title in [(axes[0], "box", "(a) Box prompt"), (axes[1], "click", "(b) Single-click prompt")]:
    for k, (i, (lab, c, mk)) in enumerate(INIT.items()):
        v = [T["overall"][f"ours_{i}{SUF[n]}|{mode}"]["iou"] for n in N]
        x = [n * (1 + 0.05 * (k - 1)) for n in N]                       # small horizontal offset so bars don't overlap
        ax.errorbar(x, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], color=c, marker=mk,
                    ms=5, lw=1.6, capsize=2, label=lab)
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N]); ax.minorticks_off()
    ax.set_xlabel("Labelled images used for fine-tuning (log scale)"); ax.set_title(title, loc="left")
axes[0].set_ylabel("Held-out mIoU (95% CI)"); axes[0].set_ylim(0, 0.65); axes[1].set_ylim(0, 0.35)
axes[0].legend(fontsize=8, loc="upper left")
ax = axes[2]
for b, lab, c, off in [("random", "SAMI − random", "#8a8986", 0.95), ("mae", "SAMI − MAE", "#eda100", 1.05)]:
    for n in N:
        d, lo, hi = T["paired"][f"ours_sami{SUF[n]}-ours_{b}{SUF[n]}"]["box"]
        ax.plot([n * off] * 2, [lo, hi], color=c, lw=2, solid_capstyle="round")
        ax.plot(n * off, d, "o", color=c, ms=6, mec="white", mew=1, label=lab if n == 100 else None)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N]); ax.minorticks_off()
ax.set_xlabel("Labelled images used for fine-tuning (log scale)"); ax.set_ylabel("Paired difference in box mIoU (95% CI)")
ax.set_title("(c) SAMI advantage on boxes", loc="left"); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); print("saved")
