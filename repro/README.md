# `repro/`: EfficientSAM reproduction code

Our re-implementation of **SAMI pretraining** (Xiong et al., *EfficientSAM*, CVPR 2024), scaled down to our
COCO subset and one Colab T4 GPU, plus the evaluation code used for every model in our report.

* **Run it:** `notebooks/EfficientSAM_Reproduction_Colab.ipynb` (step by step, resumable, on a free T4 GPU).
* **Our results:** `output/repro_results/` (per-object scores, summary tables, figures, logs).

## Pipeline

| Step | Module | What it does |
|---|---|---|
| Prompts | `make_eval_prompts.py` | Freezes one box and one click (distance-transform maximum) per held-out object: 1,620 objects in 199 images |
| Teacher | `teacher_features.py` | Runs frozen SAM ViT-H on the 800 training images + mirrored copies at 1024 px; pools 64×64 features to 16×16 |
| Pretrain | `pretrain.py`, `sami.py` | ViT-Tiny student, 75% masking, cross-attention decoder, linear projection. `--target sami` (SAM features, MSE over all tokens) or `--target mae` (normalised pixels, masked tokens) |
| Fine-tune | `finetune.py` | EfficientSAM-Ti architecture at 256 px, box or click prompts, SAM loss. `--init sami/mae/random`, `--freeze_encoder`, `--epochs`, `--lr_encoder` |
| Evaluate | `evaluate_model.py` | One model per process: mIoU, Dice, Boundary IoU, best-of-3 oracle, encoder/decoder time, memory |
| Summarise | `summarize_results.py` | Tables by size, supercategory, category-macro; paired differences with 95% bootstrap CIs over images |

All scripts are run from the repository root as modules, e.g. `python -m repro.finetune --init sami ...`, and read
three environment variables: `COCO_DATA_ROOT` (our COCO subset), `REPRO_OUT` (results folder) and optionally
`EFFSAM_REPO` (repository root, for `efficient_sam/`).

## Which code is ours

| Component | Source | Status |
|---|---|---|
| `efficient_sam/` (encoder, prompt encoder, mask decoder) | Official EfficientSAM code | Reused unchanged |
| `segment_anything`, `mobile_sam` packages | Official SAM / MobileSAM code | Reused unchanged (teacher and baselines) |
| `common.py: build_student_sam()` | Official `build_efficient_sam()` | Adapted: input size 1024 → 256, no checkpoint |
| `finetune.py: sam_loss()` | Our earlier `from_scratch/tinysam` losses | Adapted: vectorised, padding mask |
| `teacher_features.py`, `sami.py`, `pretrain.py`, rest of `finetune.py` | – | Written by our team |
| `make_eval_prompts.py`, `evaluate_model.py`, `summarize_results.py` | – | Written by our team |

## Main results (T4 GPU, 1,620 held-out objects, mIoU)

| Model | Box | Click |
|---|---|---|
| SAM ViT-H (released) | 0.780 | 0.591 |
| EfficientSAM-S (released) | 0.764 | 0.598 |
| SAM ViT-B (released) | 0.764 | 0.546 |
| EfficientSAM-Ti (released) | 0.751 | 0.501 |
| MobileSAM (released) | 0.735 | 0.535 |
| Ours, SAMI start, full fine-tuning | 0.497 | 0.268 |
| Ours, MAE start, full fine-tuning | 0.510 | 0.269 |
| Ours, random start, full fine-tuning | 0.513 | 0.275 |
| Ours, SAMI start, frozen encoder | 0.487 | 0.257 |
| Ours, MAE start, frozen encoder | 0.482 | 0.226 |
| Ours, random start, frozen encoder | 0.504 | 0.231 |

With the encoder frozen, SAMI beats MAE on single clicks by +0.031 [95% CI +0.019, +0.043]; after full fine-tuning
on 800 images the difference disappears. Full tables, confidence intervals and timings:
`output/repro_results/gpu_t4/results_summary.md`.

## Trained weights
Checkpoints are not stored in git (≈40 MB each). Link: **TO DO** (Google Drive / GitHub release).
Load a fine-tuned student with:
```python
import torch
from repro.common import build_student_sam
model = build_student_sam()
model.load_state_dict(torch.load("model_final.pt", map_location="cpu")["model"])
model.eval()   # input: RGB image resized to 256x256, values in [0, 1]
```

## History
* First version: failed on the GPU under fp16 mixed precision (`scatter` dtype error in `sami.py`); fixed with
  `torch.promote_types` before any result was produced.
* `--freeze_encoder` was added later for the ablation; it does not change runs made without it.
* This version also adds the run names for Experiment C (`*_short`, `*_lowlr`) to the evaluation and summary
  scripts. The training code is the same code that produced the results in `output/repro_results/`.

Earlier versions are not kept separately: git history records every change from here on.
