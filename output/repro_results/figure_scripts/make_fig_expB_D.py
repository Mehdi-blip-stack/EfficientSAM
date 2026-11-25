"""Dedicated figures for Experiment B (frozen encoder) and Experiment D (100 / 200 labelled images).
Same bootstrap method as repro/summarize_results.py (2,000 resamples of images, paired on the same objects)."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(sys.argv[1]); OUT = Path(sys.argv[2])
T = json.load(open(R / "results_tables.json"))
rng = np.random.default_rng(36220)
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 220, "savefig.bbox": "tight"})
INIT = {"sami": ("SAMI", "#e34948"), "mae": ("MAE", "#eda100"), "random": ("Random", "#8a8986")}


def load(m):
    return list(csv.DictReader(open(R / f"per_instance_{m}.csv")))


def paired(a, b, key):
    bb = {r["ann_id"]: float(r[key]) for r in load(b)}
    per = defaultdict(list)
    for r in load(a):
        per[r["image_id"]].append(float(r[key]) - bb[r["ann_id"]])
    s = np.array([sum(v) for v in per.values()]); c = np.array([len(v) for v in per.values()])
    idx = rng.integers(0, len(s), (2000, len(s))); bs = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def bars(ax, groups, labels, getter, ylim, ylabel=None, title=None, legend=False):
    w = 0.26; xs = np.arange(len(groups))
    for j, i in enumerate(INIT):
        v = [getter(i, g) for g in groups]
        ax.bar(xs + (j - 1) * w, [t[0] for t in v], w * 0.9, color=INIT[i][1], label=INIT[i][0] + " start",
               yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    ax.set_xticks(xs); ax.set_xticklabels(labels); ax.grid(axis="x", visible=False); ax.set_ylim(*ylim)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title, loc="left")
    if legend: ax.legend(fontsize=8, ncol=3, loc="upper left")


def dots(ax, items, title, xlabel):
    ys = np.arange(len(items))[::-1]
    for y, (lab, (d, lo, hi)) in zip(ys, items):
        sig = lo > 0 or hi < 0
        c = ("#1baf7a" if d > 0 else "#e34948") if sig else "#8a8986"
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round"); ax.plot(d, y, "o", color=c, ms=6, mec="white", mew=1)
    ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False)
    ax.set_yticks(ys); ax.set_yticklabels([l for l, _ in items]); ax.set_xlabel(xlabel); ax.set_title(title, loc="left")


# ---------------- Experiment B
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
bars(axes[0], ["small", "medium", "large"], ["small\n(n=660)", "medium\n(n=600)", "large\n(n=360)"],
     lambda i, s: T["by_size"][f"ours_{i}_frozen|click"][s], (0, 0.42), "Single-click mIoU (95% CI)",
     "(a) Encoder frozen: single clicks by object size", legend=True)
items = []
for mode in ["click", "box"]:
    for i in INIT:
        d = tuple(T["paired"]["ours_sami_frozen-ours_sami"][mode]) if i == "sami" else paired(f"ours_{i}_frozen", f"ours_{i}", f"{mode}_iou")
        items.append((f"{INIT[i][0]}, {mode}", d))
dots(axes[1], items, "(b) Cost of freezing: frozen − fully fine-tuned", "Paired difference in mIoU (95% CI)")
fig.tight_layout(); fig.savefig(OUT / "fig_r8_expB_frozen.pdf"); fig.savefig(OUT / "fig_r8_expB_frozen.png"); plt.close(fig)
print("B:", [(l, round(d[0], 3), round(d[1], 3), round(d[2], 3)) for l, d in items])

# ---------------- Experiment D
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
g = {"100": "_n100", "200": "_n200"}
bars(axes[0], ["100", "200"], ["100 labelled\nimages", "200 labelled\nimages"],
     lambda i, k: T["overall"][f"ours_{i}{g[k]}|box"]["iou"], (0, 0.38), "Held-out mIoU (95% CI)", "(a) Box prompt", legend=True)
bars(axes[1], ["100", "200"], ["100 labelled\nimages", "200 labelled\nimages"],
     lambda i, k: T["overall"][f"ours_{i}{g[k]}|click"]["iou"], (0, 0.38), None, "(b) Single-click prompt")
items = []
for k in ["100", "200"]:
    for a, b in [("sami", "random"), ("mae", "random"), ("sami", "mae")]:
        items.append((f"{k} img: {INIT[a][0]} − {INIT[b][0].lower() if b == 'random' else INIT[b][0]}", tuple(T["paired"][f"ours_{a}{g[k]}-ours_{b}{g[k]}"]["box"])))
dots(axes[2], items, "(c) Paired differences, box prompt", "Difference in box mIoU (95% CI)")
fig.tight_layout(); fig.savefig(OUT / "fig_r9_expD_few_labels.pdf"); fig.savefig(OUT / "fig_r9_expD_few_labels.png"); plt.close(fig)
print("D ok")
