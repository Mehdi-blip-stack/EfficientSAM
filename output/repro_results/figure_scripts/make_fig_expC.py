"""Figure for Experiment C: SAMI vs MAE vs random across fine-tuning regimes (click and box)."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
T = json.load(open(sys.argv[1])); out = sys.argv[2]
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 220, "savefig.bbox": "tight"})
REG = [("_frozen", "Encoder\nfrozen"), ("_short", "20 epochs"), ("_lowlr", "Encoder\nLR 1e-5"), ("", "Full\n(100 ep, LR 1e-4)")]
INIT = {"sami": ("SAMI", "#e34948"), "mae": ("MAE", "#eda100"), "random": ("Random", "#8a8986")}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
xs = np.arange(len(REG)); w = 0.26
ax = axes[0]
for j, i in enumerate(INIT):
    v = [T["overall"][f"ours_{i}{s}|click"]["iou"] for s, _ in REG]
    ax.bar(xs + (j - 1) * w, [t[0] for t in v], w * 0.9, color=INIT[i][1], label=INIT[i][0] + " start",
           yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in REG]); ax.grid(axis="x", visible=False)
ax.set_ylim(0, 0.30); ax.set_ylabel("Single-click mIoU (95% CI)"); ax.set_title("(a) Click mIoU by fine-tuning regime", loc="left")
ax.legend(fontsize=8, ncol=3, loc="upper left")
ax = axes[1]
for k, (b, lab, c, off) in enumerate([("mae", "SAMI − MAE", "#eda100", -0.12), ("random", "SAMI − random", "#8a8986", 0.12)]):
    for x, (s, _) in enumerate(REG):
        d, lo, hi = T["paired"][f"ours_sami{s}-ours_{b}{s}"]["click"]
        ax.plot([x + off, x + off], [lo, hi], color=c, lw=2, solid_capstyle="round")
        ax.plot(x + off, d, "o", color=c, ms=6, mec="white", mew=1, label=lab if x == 0 else None)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in REG]); ax.grid(axis="x", visible=False)
ax.set_ylabel("Paired difference in click mIoU (95% CI)"); ax.set_title("(b) SAMI advantage on single clicks", loc="left")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); print("saved", out)
