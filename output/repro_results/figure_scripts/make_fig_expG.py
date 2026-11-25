"""Experiment G: the SAMI student fine-tuned on 100,000 images for 5 epochs, next to the earlier SAMI students
(800 images / 100 epochs, 5,000 images / 16 and 100 epochs). 5,000 x 100 and 100,000 x 5 have the same number of steps.
Usage: python make_fig_expG.py output/repro_results/gpu_t4/results_tables.json output/repro_results/figures/fig_r11_expG_100k_images"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
T = json.load(open(sys.argv[1])); out = sys.argv[2]
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 220, "savefig.bbox": "tight"})
REG = [("ours_sami", "800 img\n100 ep"), ("ours_sami_5k", "5,000 img\n16 ep (E)"),
       ("ours_sami_5k_e100", "5,000 img\n100 ep (F)"), ("ours_sami_100k", "100,000 img\n5 ep (G)")]
COL = ["#f3b3b2", "#ec8584", "#e34948", "#a51d1c"]
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
for ax, mode, title, ylim in [(axes[0], "box", "(a) Box prompt", 0.75), (axes[1], "click", "(b) Single-click prompt", 0.45)]:
    v = [T["overall"][f"{m}|{mode}"]["iou"] for m, _ in REG]
    xs = np.arange(len(REG))
    ax.bar(xs, [t[0] for t in v], 0.62, color=COL, yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]],
           error_kw=dict(lw=0.8, capsize=3, ecolor=INK2))
    for x, t in zip(xs, v):
        ax.text(x, t[2] + 0.008, f"{t[0]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in REG], fontsize=7.5); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ylim); ax.set_title(title, loc="left")
axes[0].set_ylabel("Held-out mIoU, SAMI start (95% CI)")
ax = axes[2]
sizes = ["small", "medium", "large"]; w = 0.2
for j, (m, lab) in enumerate(REG):
    vals = [T["by_size"][f"{m}|box"][s] for s in sizes]
    vals = [x[0] if isinstance(x, list) else x for x in vals]
    ax.bar(np.arange(3) + (j - 1.5) * w, vals, w * 0.92, color=COL[j], label=lab.replace("\n", ", "))
ax.set_xticks(np.arange(3)); ax.set_xticklabels(["Small", "Medium", "Large"]); ax.grid(axis="x", visible=False)
ax.set_ylim(0, 0.85); ax.set_ylabel("Box mIoU"); ax.set_title("(c) Box mIoU by object size", loc="left")
ax.legend(fontsize=7, loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig(out + ".png"); print("saved", out + ".png")
