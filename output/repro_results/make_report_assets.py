"""Report-ready figures and tables for the Reproduction (Sec. 6) and Experiment (Sec. 7) sections.

Inputs : results_tables.json + efficiency_*.json from the T4 run (all 11 models), CPU efficiency of the
         released models (2-thread CPU run), training logs (pretraining: values printed every 10 epochs;
         fine-tuning: values printed every 5 epochs).
Outputs: figures/*.png (+ .pdf), tables.md, tables.tex
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T4 = Path(sys.argv[1]); CPU = Path(sys.argv[2]); FT_LOG = Path(sys.argv[3]); OUT = Path(sys.argv[4])
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
T = json.load(open(T4 / "results_tables.json"))
EFF = {p.stem.replace("efficiency_", ""): json.load(open(p)) for p in T4.glob("efficiency_*.json")}
CPU_EFF = {p.stem.replace("efficiency_", ""): json.load(open(p)) for p in CPU.glob("efficiency_*.json")}

NAME = {"effsam_ti": "EfficientSAM-Ti", "effsam_s": "EfficientSAM-S", "mobile_sam": "MobileSAM",
        "sam_vit_b": "SAM ViT-B", "sam_vit_h": "SAM ViT-H"}
RELEASED = ["mobile_sam", "effsam_ti", "effsam_s", "sam_vit_b", "sam_vit_h"]
COL = {"effsam_ti": "#2a78d6", "effsam_s": "#1baf7a", "mobile_sam": "#eb6834", "sam_vit_b": "#6250d6", "sam_vit_h": "#52514e"}
INIT = {"sami": ("SAMI", "#e34948"), "mae": ("MAE", "#eda100"), "random": ("Random", "#8a8986")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "savefig.dpi": 220, "savefig.bbox": "tight"})
ov = lambda m, mode, k="iou": T["overall"][f"{m}|{mode}"][k]          # (mean, lo, hi)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)


# =========================================================================================== FIGURES
# F1. Accuracy vs encoder time, released models (T4)
LABEL_POS = {"box":   {"mobile_sam": (8, -10), "effsam_ti": (-8, 8), "effsam_s": (-8, 8), "sam_vit_b": (8, -10), "sam_vit_h": (-8, 0)},
             "click": {"mobile_sam": (8, 0), "effsam_ti": (8, -8), "effsam_s": (8, 6), "sam_vit_b": (8, -8), "sam_vit_h": (-8, -10)}}
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
for ax, mode in zip(axes, ["box", "click"]):
    for m in RELEASED:
        y, lo, hi = ov(m, mode); x = EFF[m]["encoder_ms_median"]
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", ms=8, color=COL[m], mec="white", mew=1.2, capsize=3, lw=1.2, zorder=3)
        dx, dy = LABEL_POS[mode][m]
        ax.annotate(f"{NAME[m]} ({EFF[m]['params_total_M']:.0f} M)", (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="center", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_xlim(35, 5000)
    ax.set_xlabel("Encoder time per image on a T4 GPU (ms, log scale)")
    ax.set_title(f"{'Box' if mode == 'box' else 'Single-click'} prompt", loc="left")
axes[0].set_ylabel("mIoU (95% CI)"); axes[0].set_ylim(0.72, 0.795); axes[1].set_ylim(0.47, 0.62)
fig.tight_layout(); save(fig, "fig_r1_accuracy_vs_speed_released")

# F2. mIoU by COCO object size, released models
sizes = ["small", "medium", "large"]
import pandas as pd
n_size = pd.read_csv(T4 / "per_instance_effsam_ti.csv")["size"].value_counts().to_dict()
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
w = 0.16; xs = np.arange(3)
for ax, mode in zip(axes, ["box", "click"]):
    for j, m in enumerate(RELEASED):
        v = [T["by_size"][f"{m}|{mode}"][s] for s in sizes]
        ax.bar(xs + (j - 2) * w, [t[0] for t in v], w * 0.9, color=COL[m], label=NAME[m],
               yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], error_kw=dict(lw=0.8, capsize=1.5, ecolor=INK2))
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}\n(n={n_size[s]})" for s in sizes]); ax.grid(axis="x", visible=False)
    ax.set_title(f"{'Box' if mode == 'box' else 'Single-click'} prompt", loc="left"); ax.set_ylim(0, 0.9)
axes[0].set_ylabel("mIoU (95% CI)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06), fontsize=8.5)
fig.tight_layout(); save(fig, "fig_r2_by_size_released")

# F3. Reproduction: SAMI vs MAE vs random, full fine-tuning vs frozen encoder
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
w = 0.26; xs = np.arange(2)
for ax, mode in zip(axes, ["box", "click"]):
    for j, init in enumerate(["sami", "mae", "random"]):
        vals = [ov(f"ours_{init}", mode), ov(f"ours_{init}_frozen", mode)]
        bars = ax.bar(xs + (j - 1) * w, [v[0] for v in vals], w * 0.9, color=INIT[init][1], label=INIT[init][0] + " start",
                      yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v[2] + 0.006, f"{v[0]:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(["Full fine-tuning\n(encoder trained)", "Frozen encoder\n(decoder only)"])
    ax.grid(axis="x", visible=False); ax.set_title(f"{'Box' if mode == 'box' else 'Single-click'} prompt", loc="left")
    ax.set_ylim(0, 0.6 if mode == "box" else 0.33)
axes[0].set_ylabel("mIoU on 1,620 held-out objects (95% CI)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), fontsize=8.5)
fig.tight_layout(); save(fig, "fig_r3_reproduction_sami_mae_random")

# F4. Paired differences (forest plot)
PAIRS = [("effsam_s-mobile_sam", "EfficientSAM-S − MobileSAM"), ("effsam_ti-mobile_sam", "EfficientSAM-Ti − MobileSAM"),
         ("sam_vit_h-effsam_s", "SAM ViT-H − EfficientSAM-S"),
         ("ours_sami-ours_mae", "Ours, full FT: SAMI − MAE"), ("ours_sami-ours_random", "Ours, full FT: SAMI − random"),
         ("ours_sami_frozen-ours_mae_frozen", "Ours, frozen: SAMI − MAE"),
         ("ours_sami_frozen-ours_random_frozen", "Ours, frozen: SAMI − random")]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
ys = np.arange(len(PAIRS))[::-1]
for ax, mode in zip(axes, ["box", "click"]):
    for y, (k, lab) in zip(ys, PAIRS):
        d, lo, hi = T["paired"][k][mode]
        sig = lo > 0 or hi < 0
        c = ("#1baf7a" if d > 0 else "#e34948") if sig else "#8a8986"
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.plot(d, y, "o", color=c, ms=7, mec="white", mew=1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axhline(3.5, color=GRID, lw=1.2); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Difference in mIoU (95% CI, same objects)")
    ax.set_title(f"{'Box' if mode == 'box' else 'Single-click'} prompt", loc="left")
axes[0].set_yticks(ys); axes[0].set_yticklabels([lab for _, lab in PAIRS])
fig.text(0.5, -0.03, "Green / red: the 95% CI excludes zero (first model better / worse).  Grey: no significant difference.  "
         "Rows above the line: released models; below: our trained students.", ha="center", fontsize=7.5, color=INK2)
fig.tight_layout(); save(fig, "fig_r4_paired_differences")

# F5. Training curves
SAMI_PT = [1.3071, .9663, .8602, .8322, .8112, .7934, .7785, .7629, .7526, .7424, .7294, .7215, .7124, .7037, .6956, .6882, .6810,
           .6731, .6671, .6593, .6550, .6481, .6433, .6386, .6346, .6305, .6266, .6237, .6206, .6184, .6162, .6142, .6124, .6119,
           .6106, .6098, .6086, .6079, .6079, .6078, .6085]
MAE_PT = [1.3201, .8849, .8320, .8256, .8191, .8116, .8038, .7966, .7907, .7847, .7805, .7775, .7735, .7703, .7686, .7652, .7630,
          .7599, .7582, .7556, .7541, .7525, .7506, .7487, .7469, .7443, .7421, .7393, .7367, .7348, .7331, .7315, .7299, .7294,
          .7295, .7282, .7283, .7277, .7276, .7272, .7271]
assert len(SAMI_PT) == len(MAE_PT) == 41
pt_ep = list(range(1, 400, 10)) + [400]
ft = {k: ([], []) for k in ["sami", "mae", "random"]}
for line in open(FT_LOG):
    m = re.match(r"\[ft-(\w+)\] epoch (\d+)/100 loss ([\d.]+)", line.strip())
    if m:
        ft[m.group(1)][0].append(int(m.group(2))); ft[m.group(1)][1].append(float(m.group(3)))
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
axes[0].plot(pt_ep, SAMI_PT, color=INIT["sami"][1], lw=2, label="SAMI (target: SAM ViT-H features)")
axes[0].plot(pt_ep, MAE_PT, color=INIT["mae"][1], lw=2, label="MAE (target: pixels)")
axes[0].set_xlabel("Pretraining epoch"); axes[0].set_ylabel("Reconstruction loss (MSE)")
axes[0].set_title("(a) Pretraining, 800 images × 2 views", loc="left"); axes[0].legend(fontsize=8)
axes[0].text(0.98, 0.55, "Levels are not comparable:\nthe two targets differ", transform=axes[0].transAxes, ha="right", fontsize=7.5, color=INK2)
for k in ["sami", "mae", "random"]:
    axes[1].plot(*ft[k], color=INIT[k][1], lw=2, label=INIT[k][0] + " start")
axes[1].set_xlabel("Fine-tuning epoch"); axes[1].set_ylabel("Training loss (focal + Dice + IoU)")
axes[1].set_title("(b) Fine-tuning, full encoder", loc="left"); axes[1].legend(fontsize=8); axes[1].set_ylim(0.45, 1.2)
fig.tight_layout(); save(fig, "fig_r5_training_curves")

# =========================================================================================== TABLES
f3 = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
s3 = lambda t: f"{t[0]:.3f}"
dd = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
md, tex = [], []

# T1 released models
md.append("## Table R1. Released models on our 200 held-out COCO images (1,620 objects)\n")
md.append("| Model | Params (M) | Box mIoU | Click mIoU | Box Boundary IoU | Encoder ms/image, T4 | Encoder ms/image, CPU (2 threads) |\n|---|---|---|---|---|---|---|")
tex.append(r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}" "\n"
           r"\caption{\textbf{Table R1.} Released models on our 200 held-out COCO images (1,620 objects). mIoU with 95\% bootstrap CI over images.}" "\n"
           r"\begin{tabular}{lrcccrr}\toprule" "\n"
           r"Model & Params (M) & Box mIoU & Click mIoU & Box B-IoU & T4 ms & CPU ms\\\midrule")
for m in RELEASED:
    cpu = f"{CPU_EFF[m]['encoder_ms_median']:,.0f}" if m in CPU_EFF else "–"
    md.append(f"| {NAME[m]} | {EFF[m]['params_total_M']:.1f} | {f3(ov(m, 'box'))} | {f3(ov(m, 'click'))} | {s3(ov(m, 'box', 'biou'))} | "
              f"{EFF[m]['encoder_ms_median']:,.0f} | {cpu} |")
    tex.append(f"{NAME[m]} & {EFF[m]['params_total_M']:.1f} & {f3(ov(m, 'box'))} & {f3(ov(m, 'click'))} & {s3(ov(m, 'box', 'biou'))} & "
               f"{EFF[m]['encoder_ms_median']:,.0f} & {cpu}\\\\".replace(",", "{,}"))
md.append("\nT4: PyTorch fp32, batch 1, median over 196 images after 3 warm-up images. CPU: 2-thread cloud CPU; SAM ViT-B/H were not timed on CPU.")
tex.append(r"\bottomrule\end{tabular}\end{table}")

# T2 reproduction
md.append("\n## Table R2. Our reproduction: student (EfficientSAM-Ti architecture, 256 px) by pretraining and fine-tuning regime\n")
md.append("| Encoder start | Fine-tuning | Box mIoU | Click mIoU | Box Dice | Click Dice | Box Boundary IoU |\n|---|---|---|---|---|---|---|")
tex.append(r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}" "\n"
           r"\caption{\textbf{Table R2.} Our reproduction: student (EfficientSAM-Ti architecture at 256\,px), by encoder start and fine-tuning regime.}" "\n"
           r"\begin{tabular}{llccccc}\toprule" "\n"
           r"Encoder start & Fine-tuning & Box mIoU & Click mIoU & Box Dice & Click Dice & Box B-IoU\\\midrule")
for regime, suf in [("full", ""), ("frozen encoder", "_frozen")]:
    for init in ["sami", "mae", "random"]:
        m = f"ours_{init}{suf}"
        row = [INIT[init][0], regime, f3(ov(m, "box")), f3(ov(m, "click")), s3(ov(m, "box", "dice")), s3(ov(m, "click", "dice")), s3(ov(m, "box", "biou"))]
        md.append("| " + " | ".join(row) + " |"); tex.append(" & ".join(row) + r"\\")
    if suf == "":
        tex.append(r"\midrule")
tex.append(r"\bottomrule\end{tabular}\end{table}")

md.append("\n## Table R3. Paired differences in mIoU (same 1,620 objects; 95% CI; positive = first better)\n")
md.append("| Comparison | Box | Click |\n|---|---|---|")
tex.append(r"\begin{table}[H]\centering\footnotesize" "\n"
           r"\caption{\textbf{Table R3.} Paired differences in mIoU on the same objects (95\% bootstrap CI). Positive: first model better.}" "\n"
           r"\begin{tabular}{lcc}\toprule Comparison & Box & Click\\\midrule")
for k, lab in PAIRS + [("ours_sami_frozen-ours_sami", "Ours: SAMI frozen − SAMI full FT")]:
    md.append(f"| {lab} | {dd(T['paired'][k]['box'])} | {dd(T['paired'][k]['click'])} |")
    tex.append(f"{lab.replace('−', '$-$')} & {dd(T['paired'][k]['box'])} & {dd(T['paired'][k]['click'])}\\\\")
tex.append(r"\bottomrule\end{tabular}\end{table}")

# T4 paper vs ours
h = ov("sam_vit_h", "box")[0]
PAPER_AP = {"sam_vit_h": 46.5, "effsam_s": 44.4, "effsam_ti": 42.3, "mobile_sam": 38.7}
md.append("\n## Table R4. Comparison with the paper\n")
md.append("| Model | Paper: COCO AP (detector boxes) [15] | Paper: % of SAM | Ours: box mIoU (GT boxes) | Ours: % of SAM ViT-H |\n|---|---|---|---|---|")
tex.append(r"\begin{table}[H]\centering\footnotesize" "\n"
           r"\caption{\textbf{Table R4.} Our results next to the paper's. Absolute values are not comparable (different protocol); the ranking and relative gaps are.}" "\n"
           r"\begin{tabular}{lcccc}\toprule" "\n"
           r"Model & Paper COCO AP & Paper \% of SAM & Ours box mIoU & Ours \% of SAM ViT-H\\\midrule")
for m in ["sam_vit_h", "effsam_s", "effsam_ti", "mobile_sam"]:
    p = PAPER_AP[m]; o = ov(m, "box")[0]
    row = [NAME[m], f"{p:.1f}", f"{100 * p / 46.5:.1f}%", f"{o:.3f}", f"{100 * o / h:.1f}%"]
    md.append("| " + " | ".join(row) + " |"); tex.append(" & ".join(row).replace("%", r"\%") + r"\\")
tex.append(r"\midrule")
pr = [("SAMI vs MAE pretraining, 1-click COCO mIoU (ViT-Ti)", "45.5 vs 43.3 (+2.2)",
       f"frozen: {ov('ours_sami_frozen', 'click')[0]:.3f} vs {ov('ours_mae_frozen', 'click')[0]:.3f} ({T['paired']['ours_sami_frozen-ours_mae_frozen']['click'][0]:+.3f}); "
       f"full FT: {ov('ours_sami', 'click')[0]:.3f} vs {ov('ours_mae', 'click')[0]:.3f} ({T['paired']['ours_sami-ours_mae']['click'][0]:+.3f})"),
      ("EfficientSAM-S vs SAM: parameters", "about 20x fewer", f"{EFF['sam_vit_h']['params_total_M'] / EFF['effsam_s']['params_total_M']:.0f}x fewer"),
      ("EfficientSAM-S vs SAM: speed", "about 20x (A100 throughput)", f"{EFF['sam_vit_h']['encoder_ms_median'] / EFF['effsam_s']['encoder_ms_median']:.1f}x (T4 encoder latency, batch 1)")]
md.append("\n| Claim | Paper [15] | Ours |\n|---|---|---|")
for a, b, c in pr:
    md.append(f"| {a} | {b} | {c} |")
tex.append(r"\bottomrule\end{tabular}\end{table}")
tex.append(r"\begin{table}[H]\centering\footnotesize" "\n" r"\begin{tabular}{p{4.6cm}p{3.2cm}p{6.6cm}}\toprule Claim & Paper [15] & Ours\\\midrule")
for a, b, c in pr:
    tex.append(f"{a} & {b} & {c}\\\\".replace("x ", r"$\times$ ").replace("x(", r"$\times$ ("))
tex.append(r"\bottomrule\end{tabular}\end{table}")

# T5 feasibility changes
FEAS = [("SAMI teacher", "SAM ViT-H encoder", "SAM ViT-H encoder (same)"),
        ("Pretraining data", "ImageNet-1K, 1.28 M images", "800 COCO train images × 2 views (original, mirrored)"),
        ("Pretraining resolution / epochs", "224 px, 400 epochs", "256 px, 400 epochs"),
        ("SAMI target", "teacher features at full grid", "teacher features (64×64) average-pooled to the student's 16×16 grid"),
        ("Student encoder", "ViT-Tiny / ViT-Small", "ViT-Tiny (EfficientSAM-Ti architecture, 10.2 M params)"),
        ("Fine-tuning data", "SA-1B (11 M images, 1.1 B masks)", "800 COCO train images, 5,827 masks"),
        ("Fine-tuning resolution / epochs", "1024 px, 5 epochs", "256 px, 100 epochs"),
        ("Hardware", "64 A100 GPUs", "1 Colab T4 GPU (free tier)"),
        ("Evaluation", "COCO AP with ViTDet boxes; 1-click mIoU", "prompted mIoU, Dice, Boundary IoU with GT boxes and 1 click, 200 images")]
md.append("\n## Table R5. Changes made to fit our environment (template Section 6)\n")
md.append("| Aspect | Paper [15] | Ours |\n|---|---|---|")
tex.append(r"\begin{table}[H]\centering\footnotesize" "\n" r"\caption{\textbf{Table R5.} Changes made to make SAMI feasible in our environment.}" "\n"
           r"\begin{tabular}{p{3.6cm}p{4.4cm}p{6.4cm}}\toprule Aspect & Paper [15] & Ours\\\midrule")
for row in FEAS:
    md.append("| " + " | ".join(row) + " |"); tex.append(" & ".join(row).replace("×", r"$\times$") + r"\\")
tex.append(r"\bottomrule\end{tabular}\end{table}")

# T6 code ownership
CODE = [("efficient_sam/ (encoder, prompt encoder, mask decoder, EfficientSam)", "Official EfficientSAM code", "Reused unchanged"),
        ("segment_anything, mobile_sam packages", "Official SAM / MobileSAM code", "Reused unchanged (baselines only)"),
        ("repro/common.py: build_student_sam()", "Official build_efficient_sam()", "Adapted: img_size 1024 → 256, no checkpoint"),
        ("repro/finetune.py: sam_loss()", "Team's from_scratch/tinysam losses.py", "Adapted: vectorised, padding mask"),
        ("repro/teacher_features.py", "–", "Written by us: teacher pass, view alignment, pooling, caching"),
        ("repro/sami.py", "Paper Sec. 3.1, Eq. 1", "Written by us: masked encoder pass, cross-attention decoder, projection, SAMI and MAE losses"),
        ("repro/pretrain.py", "–", "Written by us: training loop, schedule, mixed precision, resume"),
        ("repro/finetune.py (rest)", "–", "Written by us: targets, prompt sampling, training loop, frozen-encoder mode"),
        ("repro/make_eval_prompts.py, evaluate_model.py, summarize_results.py", "–", "Written by us: frozen prompts, model adapters, metrics, timing, bootstrap"),
        ("EDA and preprocessing notebook", "–", "Written by us")]
md.append("\n## Table R6. Which code was reused, adapted or written by our team\n")
md.append("| Component | Source | Status |\n|---|---|---|")
tex.append(r"\begin{table}[H]\centering\footnotesize" "\n" r"\caption{\textbf{Table R6.} Which code was reused, adapted or written by our team.}" "\n"
           r"\begin{tabular}{p{5.6cm}p{3.6cm}p{5.2cm}}\toprule Component & Source & Status\\\midrule")
for row in CODE:
    md.append("| " + " | ".join(row) + " |")
    tex.append(" & ".join(r.replace("_", r"\_").replace("→", r"$\rightarrow$") for r in row) + r"\\")
tex.append(r"\bottomrule\end{tabular}\end{table}")

(OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
(OUT / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
print("\n".join(md))
