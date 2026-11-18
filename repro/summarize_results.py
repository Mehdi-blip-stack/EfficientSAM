"""Turn the per-object scores of every evaluated model into the results tables and the accuracy-vs-speed chart.

Reads   output/repro/per_instance_<model>.csv and efficiency_<model>.json (written by evaluate_model.py)
Writes  output/repro/results_summary.md, results_tables.json, fig_accuracy_vs_speed.png, fig_by_size.png

Statistics: every mean is an object-level mean (each object counts once). 95% confidence intervals come from
a bootstrap that resamples IMAGES (not objects), because objects from the same image are not independent.
Model-vs-model differences use the SAME resampled images for both models (paired bootstrap).
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(os.environ.get("REPRO_OUT", "output/repro"))
NAMES = {"effsam_ti": "EfficientSAM-Ti", "effsam_s": "EfficientSAM-S", "mobile_sam": "MobileSAM",
         "sam_vit_b": "SAM ViT-B", "sam_vit_h": "SAM ViT-H",
         "ours_sami": "Ours: SAMI init", "ours_mae": "Ours: MAE init", "ours_random": "Ours: random init",
         "ours_sami_frozen": "Ours: SAMI frozen", "ours_mae_frozen": "Ours: MAE frozen", "ours_random_frozen": "Ours: random frozen",
         "ours_sami_lowlr": "Ours: SAMI low enc. LR", "ours_mae_lowlr": "Ours: MAE low enc. LR", "ours_random_lowlr": "Ours: random low enc. LR",
         "ours_sami_short": "Ours: SAMI 20 epochs", "ours_mae_short": "Ours: MAE 20 epochs", "ours_random_short": "Ours: random 20 epochs",
         "ours_sami_n100": "Ours: SAMI 100 labelled img", "ours_mae_n100": "Ours: MAE 100 labelled img", "ours_random_n100": "Ours: random 100 labelled img",
         "ours_sami_n200": "Ours: SAMI 200 labelled img", "ours_mae_n200": "Ours: MAE 200 labelled img", "ours_random_n200": "Ours: random 200 labelled img",
         "ours_sami_5k": "Ours: SAMI 5k labelled img", "ours_mae_5k": "Ours: MAE 5k labelled img", "ours_random_5k": "Ours: random 5k labelled img"}
ORDER = ["effsam_ti", "effsam_s", "mobile_sam", "sam_vit_b", "sam_vit_h", "ours_sami", "ours_mae", "ours_random",
         "ours_sami_frozen", "ours_mae_frozen", "ours_random_frozen",
         "ours_sami_lowlr", "ours_mae_lowlr", "ours_random_lowlr", "ours_sami_short", "ours_mae_short", "ours_random_short",
         "ours_sami_n100", "ours_mae_n100", "ours_random_n100", "ours_sami_n200", "ours_mae_n200", "ours_random_n200",
         "ours_sami_5k", "ours_mae_5k", "ours_random_5k"]
N_BOOT, MIN_CAT = 2000, 10
rng = np.random.default_rng(36220)


def load(model):
    rows = list(csv.DictReader(open(OUT / f"per_instance_{model}.csv")))
    for r in rows:
        for k in r:
            if k.endswith(("iou", "dice")):
                r[k] = float(r[k])
    return rows


def boot_ci(rows, key, mask=None):
    """Object-level mean with a 95% CI from resampling images."""
    per_img = defaultdict(list)
    for i, r in enumerate(rows):
        if mask is None or mask[i]:
            per_img[r["image_id"]].append(r[key])
    sums = np.array([sum(v) for v in per_img.values()]); cnts = np.array([len(v) for v in per_img.values()])
    idx = rng.integers(0, len(sums), (N_BOOT, len(sums)))
    b = sums[idx].sum(1) / cnts[idx].sum(1)
    return sums.sum() / cnts.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5), int(cnts.sum())


def paired_diff(rows_a, rows_b, key):
    """Mean difference a - b on the same objects, CI from resampling the same images for both models."""
    b_by = {r["ann_id"]: r[key] for r in rows_b}
    per_img = defaultdict(list)
    for r in rows_a:
        per_img[r["image_id"]].append(r[key] - b_by[r["ann_id"]])
    sums = np.array([sum(v) for v in per_img.values()]); cnts = np.array([len(v) for v in per_img.values()])
    idx = rng.integers(0, len(sums), (N_BOOT, len(sums)))
    b = sums[idx].sum(1) / cnts[idx].sum(1)
    return sums.sum() / cnts.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    models = [m for m in ORDER if (OUT / f"per_instance_{m}.csv").exists()]
    data = {m: load(m) for m in models}
    eff = {m: json.load(open(OUT / f"efficiency_{m}.json")) for m in models}
    ids = [set(r["ann_id"] for r in data[m]) for m in models]
    assert all(s == ids[0] for s in ids), "models were evaluated on different objects"
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    T, L = {}, []
    w = L.append
    n_obj, n_img = len(data[models[0]]), len({r["image_id"] for r in data[models[0]]})
    w(f"# Baseline reproduction results\n\n{n_obj} non-crowd objects in {n_img} held-out COCO val2017 images; "
      f"identical frozen prompts for every model; 3 candidate masks, the one with the highest predicted IoU is scored; "
      f"masks compared at original resolution. Brackets: 95% bootstrap CI over images ({N_BOOT} resamples).\n")

    # ---- Table A: overall quality ----
    w("## A. Segmentation quality (object-level means)\n")
    w("| Model | Prompt | mIoU | Dice | Boundary IoU | Oracle mIoU (best of 3) |\n|---|---|---|---|---|---|")
    T["overall"] = {}
    for m in models:
        for mode in ["box", "click"]:
            vals = {k: boot_ci(data[m], f"{mode}_{k}") for k in ["iou", "dice", "biou", "oracle_iou"]}
            T["overall"][f"{m}|{mode}"] = {k: v[:3] for k, v in vals.items()}
            w(f"| {NAMES[m]} | {mode} | {f(vals['iou'])} | {f(vals['dice'])} | {f(vals['biou'])} | {f(vals['oracle_iou'])} |")

    # ---- Table B: by COCO size ----
    w("\n## B. mIoU by COCO object size (recomputed mask area)\n")
    sizes = ["small", "medium", "large"]
    counts = {s: sum(r["size"] == s for r in data[models[0]]) for s in sizes}
    w("| Model | Prompt | " + " | ".join(f"{s} (n={counts[s]})" for s in sizes) + " |\n|---|---|---|---|---|")
    T["by_size"] = {}
    for m in models:
        for mode in ["box", "click"]:
            v = {s: boot_ci(data[m], f"{mode}_iou", [r["size"] == s for r in data[m]]) for s in sizes}
            T["by_size"][f"{m}|{mode}"] = {s: v[s][:3] for s in sizes}
            w(f"| {NAMES[m]} | {mode} | " + " | ".join(f(v[s]) for s in sizes) + " |")

    # ---- Table C: by supercategory + category-macro ----
    supers = sorted({r["supercategory"] for r in data[models[0]]}, key=lambda s: -sum(r["supercategory"] == s for r in data[models[0]]))
    w("\n## C. Box-prompt mIoU by supercategory\n")
    w("| Supercategory (n) | " + " | ".join(NAMES[m] for m in models) + " |\n|---|" + "---|" * len(models))
    for s in supers:
        n = sum(r["supercategory"] == s for r in data[models[0]])
        w(f"| {s} ({n}) | " + " | ".join(f"{np.mean([r['box_iou'] for r in data[m] if r['supercategory'] == s]):.3f}" for m in models) + " |")
    w("\nSame table for the single-click prompt:\n")
    w("| Supercategory (n) | " + " | ".join(NAMES[m] for m in models) + " |\n|---|" + "---|" * len(models))
    for s in supers:
        n = sum(r["supercategory"] == s for r in data[models[0]])
        w(f"| {s} ({n}) | " + " | ".join(f"{np.mean([r['click_iou'] for r in data[m] if r['supercategory'] == s]):.3f}" for m in models) + " |")
    cat_n = defaultdict(int)
    for r in data[models[0]]:
        cat_n[r["category"]] += 1
    big = sorted(c for c, n in cat_n.items() if n >= MIN_CAT)
    w(f"\n## D. Category-macro mIoU (mean over the {len(big)} categories with at least {MIN_CAT} eval objects; every category counts equally)\n")
    w("| Model | box | click |\n|---|---|---|")
    T["macro"] = {}
    for m in models:
        mac = {mode: float(np.mean([np.mean([r[f"{mode}_iou"] for r in data[m] if r["category"] == c]) for c in big])) for mode in ["box", "click"]}
        T["macro"][m] = mac
        w(f"| {NAMES[m]} | {mac['box']:.3f} | {mac['click']:.3f} |")
    w(f"\nCategories with fewer than {MIN_CAT} eval objects are excluded from the macro average; categories with none "
      f"(fire hydrant, bear, snowboard, toaster) are unavailable.")

    # ---- Table E: paired differences ----
    if len(models) > 1:
        w("\n## E. Paired differences in mIoU (same objects; positive = first model better)\n")
        w("| Comparison | box | click |\n|---|---|---|")
        T["paired"] = {}
        pairs = [(a, b) for a, b in [("effsam_ti", "mobile_sam"), ("effsam_s", "mobile_sam"), ("effsam_s", "effsam_ti"),
                                     ("sam_vit_b", "effsam_s"), ("sam_vit_h", "effsam_s"),
                                     ("ours_sami", "ours_mae"), ("ours_sami", "ours_random"), ("ours_mae", "ours_random"),
                                     ("ours_sami_frozen", "ours_mae_frozen"), ("ours_sami_frozen", "ours_random_frozen"),
                                     ("ours_mae_frozen", "ours_random_frozen"), ("ours_sami_frozen", "ours_sami"),
                                     ("ours_sami_lowlr", "ours_mae_lowlr"), ("ours_sami_lowlr", "ours_random_lowlr"),
                                     ("ours_mae_lowlr", "ours_random_lowlr"), ("ours_sami_lowlr", "ours_sami"),
                                     ("ours_sami_short", "ours_mae_short"), ("ours_sami_short", "ours_random_short"),
                                     ("ours_mae_short", "ours_random_short"), ("ours_sami_short", "ours_sami"),
                                     ("ours_sami_n100", "ours_mae_n100"), ("ours_sami_n100", "ours_random_n100"),
                                     ("ours_mae_n100", "ours_random_n100"),
                                     ("ours_sami_n200", "ours_mae_n200"), ("ours_sami_n200", "ours_random_n200"),
                                     ("ours_mae_n200", "ours_random_n200"),
                                     ("ours_sami_5k", "ours_mae_5k"), ("ours_sami_5k", "ours_random_5k"),
                                     ("ours_mae_5k", "ours_random_5k"), ("ours_sami_5k", "ours_sami")]
                 if a in models and b in models]
        for a, b in pairs:
            d = {mode: paired_diff(data[a], data[b], f"{mode}_iou") for mode in ["box", "click"]}
            T["paired"][f"{a}-{b}"] = d
            w(f"| {NAMES[a]} − {NAMES[b]} | {d['box'][0]:+.3f} [{d['box'][1]:+.3f}, {d['box'][2]:+.3f}] | "
              f"{d['click'][0]:+.3f} [{d['click'][1]:+.3f}, {d['click'][2]:+.3f}] |")

    # ---- Table F: efficiency ----
    e0 = eff[models[0]]
    w(f"\n## F. Efficiency ({e0['device'].upper()}, {e0['threads']} threads, PyTorch {e0['torch']}, batch size 1)\n")
    w("| Model | Params total (M) | Params encoder (M) | Encoder ms/image (median) | Decoder ms/prompt (median) | Peak process memory (MB) |\n|---|---|---|---|---|---|")
    for m in models:
        e = eff[m]
        w(f"| {NAMES[m]} | {e['params_total_M']:.2f} | {e['params_encoder_M']:.2f} | {e['encoder_ms_median']:.0f} | "
          f"{e['decoder_ms_per_prompt_median']:.0f} | {e['peak_memory_MB']:.0f} |")
    w("\nEncoder time is one forward pass per image (first 3 images excluded as warm-up). Decoder time includes "
      "upsampling the masks to the original resolution. Peak memory is the whole Python process (model, image, masks).")
    T["efficiency"] = eff
    (OUT / "results_summary.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(T, open(OUT / "results_tables.json", "w"), indent=1, default=float)

    # ---- Figures ----
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200})
    col = dict(zip(ORDER, ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#52514e", "#e34948", "#eda100", "#888888", "#a02c2c", "#a87200", "#444444", "#f08a89", "#f5c95c", "#bbbbbb", "#c45a5a", "#c99a2e", "#666666", "#ff7b7a", "#ffd166", "#cfcfcf", "#d9534f", "#e0b040", "#999999", "#b22222", "#c08000", "#555555"]))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for a, mode in zip(ax, ["box", "click"]):
        for m in models:
            v = T["overall"][f"{m}|{mode}"]["iou"]; x = eff[m]["encoder_ms_median"] / 1000
            a.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=7, color=col[m], capsize=3)
            a.annotate(f"{NAMES[m]}\n{eff[m]['params_total_M']:.1f} M params", (x, v[0]), xytext=(8, -4), textcoords="offset points", fontsize=8)
        a.set_xscale("log"); a.set_xlabel(f"Encoder time per image, s ({e0['device'].upper()}, log scale)")
        a.set_title(f"{'Box' if mode == 'box' else 'Single-click'} prompt", loc="left", fontweight="bold")
        a.margins(x=0.35, y=0.25)
    ax[0].set_ylabel("mIoU (95% CI)")
    fig.tight_layout(); fig.savefig(OUT / "fig_accuracy_vs_speed.png")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    xs = np.arange(3); wd = 0.8 / len(models)
    for a, mode in zip(ax, ["box", "click"]):
        for j, m in enumerate(models):
            v = [T["by_size"][f"{m}|{mode}"][s] for s in sizes]
            a.bar(xs + (j - (len(models) - 1) / 2) * wd, [t[0] for t in v], wd * 0.92, color=col[m], label=NAMES[m],
                  yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], error_kw=dict(lw=0.8, capsize=2))
        a.set_xticks(xs); a.set_xticklabels([f"{s}\n(n={counts[s]})" for s in sizes]); a.grid(axis="x", visible=False)
        a.set_title(f"{'Box' if mode == 'box' else 'Single-click'} prompt", loc="left", fontweight="bold")
    ax[0].set_ylabel("mIoU (95% CI)"); ax[0].set_ylim(0, 1.0)
    ax[0].legend(fontsize=8, loc="upper left", ncol=len(models), columnspacing=1.0, handlelength=1.2)
    fig.tight_layout(); fig.savefig(OUT / "fig_by_size.png")
    print("\n".join(L))


if __name__ == "__main__":
    main()
