"""Experiment F: 5,000 images trained 16 vs 100 epochs, next to the 800-image runs.
Usage: python make_fig_expF.py output/repro_results/gpu_t4/results_tables.json output/repro_results/figures/fig_r10_expF_longer_training"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
T = json.load(open(sys.argv[1])); out = sys.argv[2]
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 220, "savefig.bbox": "tight"})
REG = [("", "800 images\n100 epochs"), ("_5k", "5,000 images\n16 epochs (E)"), ("_5k_e100", "5,000 images\n100 epochs (F)")]
INIT = {"sami": ("SAMI", "#e34948"), "mae": ("MAE", "#eda100"), "random": ("Random", "#8a8986")}
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.05]})
xs = np.arange(len(REG)); w = 0.26
for ax, mode, title, ylim in [(axes[0], "box", "(a) Box prompt", 0.72), (axes[1], "click", "(b) Single-click prompt", 0.42)]:
    for j, i in enumerate(INIT):
        v = [T["overall"][f"ours_{i}{s}|{mode}"]["iou"] for s, _ in REG]
        ax.bar(xs + (j - 1) * w, [t[0] for t in v], w * 0.9, color=INIT[i][1], label=INIT[i][0] + " start",
               yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in REG], fontsize=8); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ylim); ax.set_title(title, loc="left")
axes[0].set_ylabel("Held-out mIoU (95% CI)"); axes[0].legend(fontsize=8, ncol=3, loc="upper left")
ax = axes[2]
for b, lab, c, off in [("random", "SAMI − random", "#8a8986", -0.1), ("mae", "SAMI − MAE", "#eda100", 0.1)]:
    for x, (s, _) in enumerate(REG):
        d, lo, hi = T["paired"][f"ours_sami{s}-ours_{b}{s}"]["box"]
        ax.plot([x + off] * 2, [lo, hi], color=c, lw=2, solid_capstyle="round")
        ax.plot(x + off, d, "o", color=c, ms=6, mec="white", mew=1, label=lab if x == 0 else None)
ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(xs); ax.set_xticklabels([l for _, l in REG], fontsize=8)
ax.grid(axis="x", visible=False); ax.set_ylabel("Paired difference in box mIoU (95% CI)")
ax.set_title("(c) SAMI advantage on boxes", loc="left"); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(out + ".png"); print("saved", out + ".png")
